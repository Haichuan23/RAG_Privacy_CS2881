from modules.TogetherAI_API import chat_completion

import math
import torch
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, pipeline


class LM(object):
    def __init__(self, my_args, llm_args) -> None:
        self.api = my_args.api
        self.is_chat_model = llm_args.is_chat_model
        if llm_args.together_ckpt is not None:
            self.model_name = llm_args.together_ckpt.split("/")[-1]
        else:
            self.model_name = llm_args.hf_ckpt.split("/")[-1]
        
        if my_args.api == 'hf':
            self.tokenizer = AutoTokenizer.from_pretrained(llm_args.hf_ckpt)
            self.model = AutoModelForCausalLM.from_pretrained(llm_args.hf_ckpt, device_map='auto').eval()
    
            self.model.resize_token_embeddings(len(self.tokenizer))
                
            self.generation_config = GenerationConfig(
                max_new_tokens=llm_args.max_new_tokens,
                do_sample=llm_args.do_sample,
                temperature=llm_args.temperature,
                top_p=llm_args.top_p,
                top_k=llm_args.top_k,
                num_beams=llm_args.num_beams,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            )
            
            self.loss_fn = CrossEntropyLoss(reduction="none")
        elif my_args.api == 'together':
            assert llm_args.together_ckpt is not None
            support = ['llama', 'falcon', 'alpaca', 'vicuna', 'mistral', 'mixtral', 'solar', 'yi', 'platypus', 'capybara', 'wizardlm', 'qwen']
            assert any((s in llm_args.hf_ckpt.lower() and s in llm_args.together_ckpt.lower()) for s in support)
            
            self.tokenizer = AutoTokenizer.from_pretrained(llm_args.hf_ckpt)
            
            self.generation_config = {
                "model_ckpt": llm_args.together_ckpt,
                "max_tokens": llm_args.max_new_tokens,
                "temperature": llm_args.temperature,
                "top_k": llm_args.top_k,
                "top_p": llm_args.top_p,
                "stop": llm_args.stop_tokens
            }
            if llm_args.is_chat_model:
                self.generation_config["system_prompt"] = llm_args.system_prompt
        else:
            raise NotImplementedError
        
        assert self.tokenizer is not None
        
    def generate(self, lm_input: str, compute_generation_scores=False, compute_input_loss=False):
        """input a string, output a string"""

        output_dict = dict()

        if self.api == 'hf':
            assert torch.cuda.is_available(), "CUDA is not available???"

            # Apply chat template if this is a chat model
            if self.is_chat_model and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                print("\n" + "="*80)
                # print("DEBUG: Applying chat template")
                # print(f"Original input length: {len(lm_input)} chars")
                # print(f"Original input (first 200 chars): {lm_input[:200]}...")

                # For RAG, the input might be: [retrieved_docs]\n\n[query]
                # We want to present this naturally to the chat model
                # Option: Just put everything in user message (simple approach for now)
                messages = [{"role": "user", "content": lm_input}]
                formatted_input = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # print(f"\nFormatted input length: {len(formatted_input)} chars")
                # print(f"Formatted input (first 500 chars):\n{formatted_input[:500]}...")
                # print(f"Formatted input (last 200 chars):\n...{formatted_input[-200:]}")

                inputs = self.tokenizer(formatted_input, return_tensors="pt")
            else:
                print("\n" + "="*80)
                print("DEBUG: Using raw input (no chat template)")
                print(f"Input length: {len(lm_input)} chars")
                print(f"Input (first 200 chars): {lm_input[:200]}...")
                inputs = self.tokenizer(lm_input, return_tensors="pt")

            input_ids = inputs["input_ids"].cuda()  #! [1, *]
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.cuda()

            print(f"\nInput token count: {input_ids.shape[1]}")
            print(f"Has attention mask: {attention_mask is not None}")
            assert input_ids.ndim == 2 and input_ids.shape[0] == 1

            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                output_ids = generation_output.sequences[0]
                generated_tokens = output_ids[input_ids.shape[1]:]  #! truncate, only output the LLM response

                print(f"\nGenerated token count: {len(generated_tokens)}")
                print(f"Generated token IDs (first 20): {generated_tokens[:20].tolist()}")
                
                if compute_generation_scores:
                    # compute transition scores: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores 
                    # discussion: https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075 
                    # to get probability from score, just use: exp(score)
                    transition_scores = self.model.compute_transition_scores(
                        generation_output.sequences, generation_output.scores, normalize_logits=True
                    )
                    assert generated_tokens.shape == transition_scores[0].shape
                    generation_scores = transition_scores[0]
                    output_dict["generation_scores"] = generation_scores
                    output_dict["generation_probs"] = torch.exp(generation_scores)
                
                if compute_input_loss:
                    forward_output = self.model(
                        input_ids=input_ids, labels=input_ids
                    )
                    logits = forward_output.logits  #! [1, *, 50257]
                    logits_shift_left = logits[:, :-1, :]
                    logits_shift_left = logits_shift_left.reshape(-1, logits_shift_left.size(-1))
                    labels = input_ids[:, 1:]
                    labels = labels.reshape(-1)
                    token_losses = self.loss_fn(logits_shift_left, labels)
                    output_dict["token_loss_list"] = token_losses.tolist()
                    output_dict["total_input_loss"] = sum(output_dict["token_loss_list"]) / len(output_dict["token_loss_list"])
                    # perplexity: https://huggingface.co/docs/transformers/perplexity 
                    output_dict["token_ppl_list"] = torch.exp(token_losses).tolist()
                    output_dict["total_input_ppl"] = math.exp(output_dict["total_input_loss"])
            
            lm_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            print(f"\nDecoded output length: {len(lm_output)} chars")
            print(f"Decoded output (first 200 chars): {lm_output[:200]}")
            if len(lm_output) > 200:
                print(f"Decoded output (last 200 chars): ...{lm_output[-200:]}")
            print("="*80 + "\n")

            output_dict["lm_output"] = lm_output
        elif self.api == 'together':
            if self.is_chat_model:
                lm_output = chat_completion(prompt=lm_input, **self.generation_config)
            else:
                raise NotImplementedError
                # lm_output = text_completion_TogetherAI(prompt=lm_input + "\n\nSure, here's the text before \"Here is a sentence: \": ", **self.generation_config)
            output_dict["lm_output"] = lm_output
        else:
            raise NotImplementedError
        
        return output_dict