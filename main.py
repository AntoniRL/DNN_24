import argparse
import json
import os
import time
from signal_reader import SignalReader
from record_evaluator import RecordEvaluator

db_dir_in_docker = 'D:/05_Studia/Semestr_9_mgr/dnn/BazyDanych/mit-bih-atrial-fibrillation-database-1.0.0/files'

def main(save_result_dir, records):
    print(f'no of records: {len(records)}')
    eval_start = time.time()
    for record_file_name in records:
        print(f'processing record: {record_file_name}')
        start = time.time()
        signal_reader = SignalReader(os.path.join(db_dir_in_docker, record_file_name))

        record_eval = RecordEvaluator(save_result_dir)
        record_eval.evaluate(signal_reader)
        print(f'took: {time.time() - start} seconds')
    print(f'full eval took {time.time() - eval_start} seconds')




if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("save_result_dir")
    arg_parser.add_argument("files_to_eval_list_file_path")
    args = arg_parser.parse_args()
    with open(args.files_to_eval_list_file_path, 'r') as f:
        records = json.load(f)
    main(args.save_result_dir, records)