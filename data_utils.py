from torch.utils.data import Dataset
import torch
import copy


def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data":
            batch[n] = batch[n].to(gpuid)


class PreferenceBaseDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=2048, is_test=False, insert_eos=False):
        """data format: article, abstract, [(candidiate_i, score_i)]"""
        self.data = data
        self.tokenizer = copy.deepcopy(tokenizer)
        self.max_len = max_len
        self.is_test = is_test
        self.num = len(self.data)
        self.insert_eos = insert_eos

    def __len__(self):
        return self.num

    def encode_with_messages_format(self, example):
        """
        from https://github.com/allenai/open-instruct/blob/main/open_instruct/dpo_tune.py#L252
        Here we assume each example has a rejected and chosen field, both of which are a list of messages.
        Each message is a dict with 'role' and 'content' fields.
        We concatenate all messages with the roles as delimiters and tokenize them together.
        We assume only the last message is different, and the prompt is contained in the list of messages.
        """
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]
        if len(chosen_messages) == 0:
            raise ValueError("chosen messages field is empty.")
        if len(rejected_messages) == 0:
            raise ValueError("rejected messages field is empty.")
        eos_insert = self.tokenizer.eos_token if self.insert_eos else ""
        def _concat_messages(messages):
            message_text = ""
            for message in messages:
                if message["role"] == "system":
                    message_text += "<|system|>\n" + message["content"].strip() + eos_insert + "\n"
                elif message["role"] == "user":
                    message_text += "<|user|>\n" + message["content"].strip() + eos_insert + "\n"
                elif message["role"] == "assistant":
                    message_text += (
                        "<|assistant|>\n"
                        + message["content"].strip()
                        + self.tokenizer.eos_token
                        + "\n"
                    )
                else:
                    raise ValueError("Invalid role: {}".format(message["role"]))
            return message_text

        def encode_messages(messages):
            example_text = _concat_messages(messages).strip()
            tokenized_example = self.tokenizer(
                example_text,
                return_tensors="pt",
                max_length=self.max_len,
                truncation=True,
            )
            input_ids = tokenized_example.input_ids
            masks = torch.ones_like(input_ids)

            # mask the non-assistant part for avoiding loss
            for message_idx, message in enumerate(messages):
                if message["role"] != "assistant":
                    if message_idx == 0:
                        message_start_idx = 0
                    else:
                        message_start_idx = self.tokenizer(
                            _concat_messages(messages[:message_idx]),
                            return_tensors="pt",
                            max_length=self.max_len,
                            truncation=True,
                        ).input_ids.shape[1]
                    if (
                        message_idx < len(messages) - 1
                        and messages[message_idx + 1]["role"] == "assistant"
                    ):
                        # here we also ignore the role of the assistant
                        messages_so_far = (
                            _concat_messages(messages[: message_idx + 1])
                            + "<|assistant|>\n"
                        )
                    else:
                        messages_so_far = _concat_messages(messages[: message_idx + 1])
                    message_end_idx = self.tokenizer(
                        messages_so_far,
                        return_tensors="pt",
                        max_length=self.max_len,
                        truncation=True,
                    ).input_ids.shape[1]
                    masks[:, message_start_idx:message_end_idx] = 0

                    if message_end_idx >= self.max_len:
                        break

            return {
                "input_ids": input_ids.flatten(),
                "masks": masks.flatten(),
            }

        chosen_encoded = encode_messages(chosen_messages)
        rejected_encoded = encode_messages(rejected_messages)

        return {
            "chosen_input_ids": chosen_encoded["input_ids"],
            "chosen_masks": chosen_encoded["masks"],
            "rejected_input_ids": rejected_encoded["input_ids"],
            "rejected_masks": rejected_encoded["masks"],
        }

    def __getitem__(self, idx):
        data = self.data[idx]
        encoded = self.encode_with_messages_format(data)
        if self.is_test:
            encoded["data"] = data
        return encoded


def collate_preference_base(batch, pad_token_id, is_test=False):
    def pad(X, padding, max_len=-1, pad_side="left"):
        assert pad_side in ["left", "right"]
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * padding
        attention_mask = torch.zeros(len(X), max_len, dtype=X[0].dtype)
        for i, x in enumerate(X):
            if pad_side == "left":
                result[i, -x.size(0) :] = x
                attention_mask[i, -x.size(0) :] = 1
            else:
                result[i, : x.size(0)] = x
                attention_mask[i, : x.size(0)] = 1
        return result, attention_mask

    # pad chosen
    chosen_input_ids, chosen_attention_mask = pad(
        [x["chosen_input_ids"] for x in batch], pad_token_id, pad_side="left"
    )
    chosen_masks, _ = pad([x["chosen_masks"] for x in batch], 0, pad_side="left")

    # pad rejected
    rejected_input_ids, rejected_attention_mask = pad(
        [x["rejected_input_ids"] for x in batch], pad_token_id, pad_side="left"
    )
    rejected_masks, _ = pad([x["rejected_masks"] for x in batch], 0, pad_side="left")

    # concatenate
    input_ids = torch.unbind(chosen_input_ids) + torch.unbind(rejected_input_ids)
    attention_mask = torch.unbind(chosen_attention_mask) + torch.unbind(rejected_attention_mask)
    masks = torch.unbind(chosen_masks) + torch.unbind(rejected_masks)

    # right pad now
    input_ids, _attention_mask = pad(input_ids, pad_token_id, pad_side="right")
    attention_mask, _ = pad(attention_mask, 0, pad_side="right")
    attention_mask = attention_mask * _attention_mask
    masks, _ = pad(masks, 0, pad_side="right")

    result = {
        "input_ids": input_ids,
        "masks": masks,
        "attention_mask": attention_mask,
    }
    if is_test:
        result["data"] = [x["data"] for x in batch]
        result["chosen_input_ids"] = [x["chosen_input_ids"] for x in batch]
        result["rejected_input_ids"] = [x["rejected_input_ids"] for x in batch]
    return result


class PreferenceDataset(PreferenceBaseDataset):
    def __getitem__(self, idx):
        data = self.data[idx]
        encoded = self.encode_with_messages_format(data)
        encoded["chosen_logprob"] = data["chosen_logprob"]
        encoded["rejected_logprob"] = data["rejected_logprob"]
        if self.is_test:
            encoded["data"] = data
        return encoded

    
def collate_preference(batch, pad_token_id, is_test=False):
    results = collate_preference_base(batch, pad_token_id, is_test=is_test)
    chosen_logprob = torch.tensor([x["chosen_logprob"] for x in batch])
    rejected_logprob = torch.tensor([x["rejected_logprob"] for x in batch])
    results["chosen_logprob"] = chosen_logprob
    results["rejected_logprob"] = rejected_logprob
    return results