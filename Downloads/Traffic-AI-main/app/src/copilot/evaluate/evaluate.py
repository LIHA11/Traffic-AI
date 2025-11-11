import os
from datetime import datetime
import asyncio
from typing import List, Tuple, Callable
import pandas as pd

from src.conversations.vo.message import Message
from src.copilot.evaluate.llm_as_judge import evaluate_by_llm
from src.copilot.copilot_v3 import CopilotAgentRuntime

async def evaluate(
    runtime: CopilotAgentRuntime,
    eval_set: List[Tuple[List[Message], str]], 
    n: int = 4,
    batch_size: int = 5
) -> Tuple[float, int]:
    all_pred, all_runtime = [], []
    total_batches = (len(eval_set) + batch_size - 1) // batch_size

    print(f"[INFO] Starting evaluation: {len(eval_set)} samples, {n} trials, batch size {batch_size}")

    # Run n trials
    for trial in range(n):
        pred_with_runtime = []
        print(f"[INFO] Starting trial {trial + 1}/{n}")
        for batch_idx, batch_start in enumerate(range(0, len(eval_set), batch_size)):
            batch = eval_set[batch_start:batch_start + batch_size]
            print(f"[INFO]  Trial {trial+1}: Processing batch {batch_idx + 1}/{total_batches} "
                  f"(samples {batch_start + 1}-{min(batch_start + batch_size, len(eval_set))})...")
            preds = await asyncio.gather(*(runtime.create_until_finish(messages) for messages, _ in batch))
            pred_with_runtime.extend(preds)
        print(f"[INFO] Trial {trial + 1} completed.")

        # Unpack predictions and runtimes for this trial
        all_pred.append([p[0].content for p in pred_with_runtime])
        all_runtime.append([round(p[1]) for p in pred_with_runtime])

    # Transpose trials to get per-sample lists
    transposed_pred = list(map(list, zip(*all_pred)))
    transposed_runtime = list(map(list, zip(*all_runtime)))

    records = []
    num_eval_samples = 0

    print(f"\n[INFO] Evaluating predictions with LLM judge for {len(eval_set)} samples...")
    for idx, (messages, ans) in enumerate(eval_set):
        if ans is None:
            print(f"[WARN] Sample {idx + 1}: No reference answer, skipping.")
            continue
        preds = transposed_pred[idx]
        runtimes = transposed_runtime[idx]

        print(f"[INFO]  Evaluating sample {idx + 1}/{len(eval_set)} with LLM judge...")
        judge_results = await asyncio.gather(
            *(evaluate_by_llm(prediction=pred, ans=ans) for pred in preds)
        )
        correctness = [int(result[0]) for result in judge_results]
        judge_explanations = [result[1] for result in judge_results]
        stability_accuracy = sum(correctness) / n

        record = {
            'user_message': messages[0].content,
            'expected_ans': ans,
            'stability_accuracy': stability_accuracy,
        }
        for i in range(n):
            record[f'prediction_{i+1}'] = preds[i]
            record[f'runtime_{i+1} (s)'] = runtimes[i]
            record[f'is_correct_{i+1}'] = correctness[i]
            record[f'judge_explanation_{i+1}'] = judge_explanations[i]
        records.append(record)
        num_eval_samples += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(eval_set):
            print(f"[INFO]  Progress: {idx + 1}/{len(eval_set)} samples evaluated.")

    if num_eval_samples == 0:
        print("[WARN] No valid samples to evaluate.")
        return 0.0, 0

    result_df = pd.DataFrame(records)
    overall_accuracy = result_df['stability_accuracy'].mean()

    print(f"\n[RESULT] Evaluation completed. Overall stability accuracy: {overall_accuracy:.4f}")
    print("[INFO] Sample evaluation results:")
    print(result_df.head())

    os.makedirs('./verification/eval_results/', exist_ok=True)
    csv_path = f'./verification/eval_results/llm_judge_results_{datetime.now():%Y%m%d%H%M%S}.csv'
    try:
        result_df.to_csv(csv_path, index=False)
        print(f"[INFO] Results exported to CSV: {csv_path}")
    except Exception as e:
        print(f"[ERROR] Failed to export results to CSV: {e}")
        raise RuntimeError(f"Failed to export results to CSV: {e}")

    return overall_accuracy, num_eval_samples