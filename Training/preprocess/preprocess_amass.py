import os
import joblib
import numpy as np
import argparse

def process_files(directory):
    """
    Finds the npz files in the directory and processes them ready to train EmbodiedPose
    """
    result = {}
    num_success = 0
    for root,_, files in os.walk(directory):
        filenames = [os.path.join(root, f) for f in files]
        for file in filenames:
            if file.endswith('npz'):
                data = np.load(file)
                if all(key in data.files for key in ['trans', 'betas', 'gender', 'root_orient', 'pose_body']):
                    pose_aa = np.concatenate([data['root_orient'], data['pose_body']], axis=-1)
                    result[file] = {
                            'trans': data['trans'],
                            'beta':data['betas'],
                            'gender':data['gender'],
                            'pose_aa': pose_aa
                            }
                    num_success = num_success + 1
                else:
                    if all(key in data.files for key in ['trans', 'betas', 'gender', 'poses']):
                        # need to extract the things from above
                        # root_orient is the first three columns of poses already
                        # pose_body is also right next to that
                        result[file] = {
                                'trans': data['trans'],
                                'beta':data['betas'],
                                'gender':data['gender'],
                                'pose_aa': data['poses'][:, :66]
                        }
                        num_success = num_success + 1

    print(f"Found {num_success} motions!")
    return result


def save_results(results, out_path):
    with open(out_path, 'wb') as f:
        joblib.dump(results, f)
    print("Saved results to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="path/to/ammass_datasets")
    parser.add_argument("--output", default="path/to/processed_data.pkl")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Argument --input_dir={args.input_dir} does not exist")
        exit(1)
    if not os.path.exists(os.path.dirname(args.output)):
        print(f"Containing folder of --output={args.output} does not exist")
        exit(1)

    results = process_files(args.input_dir)
    save_results(results, args.output)



