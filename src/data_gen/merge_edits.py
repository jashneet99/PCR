import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', "--dir", type=str, default='data')
    parser.add_argument('-d', '--dataset', type=str, default='comve', choices=['comve', 'ecqa', 'esnli'])
    args = parser.parse_args()
    
    dir = args.dir
    dataset_name = args.dataset
    
    edits_gen_org = os.path.join(dir, 'counterfactual', dataset_name, "gen_org.json")
    with open(edits_gen_org, 'r') as f:
        gen_org = json.load(f)

    edits_ext_org = os.path.join(dir, 'counterfactual', dataset_name, f"ext_org.json")
    with open(edits_ext_org, 'r') as f:
        ext_org = json.load(f)

    edits_gen_final = os.path.join(dir, 'counterfactual', dataset_name, f"gen_final.json")
    output_dir = os.path.dirname(edits_gen_final)
    os.makedirs(output_dir, exist_ok=True)

    # Only attempt merging if there were failures AND gen_failed.json exists
    edits_gen_failed = os.path.join(dir, 'counterfactual', dataset_name, f"gen_failed.json")
    if ext_org['num_failed'] != 0 and os.path.exists(edits_gen_failed):
        with open(edits_gen_failed, 'r') as f:
            gen_failed = json.load(f)

        for data in gen_org:
            if data['idx'] in ext_org['extract_edits_failed_idx']:
                for new_data in gen_failed:
                    if data['idx'] == new_data['idx']:
                        data.update(new_data)
        print(f"Merged {ext_org['num_failed']} failed records from gen_failed.json")
    else:
        print("No failed records — copying gen_org.json directly to gen_final.json")

    with open(edits_gen_final, 'w') as f:
        json.dump(gen_org, f, indent=4)

    print(f"Saved gen_final.json with {len(gen_org)} records")

main()