from data_format.format_esnli import format_esnli
from data_format.format_ecqa import format_ecqa
from data_format.format_comve import format_comve


def format_dataset(dataset_name: str, split: str):
    if dataset_name == "esnli":
        return format_esnli(split)
    elif dataset_name == "ecqa":
        return format_ecqa(split)
    elif dataset_name == "comve":
        return format_comve(split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def format_all():
    datasets = ['esnli', 'comve', 'ecqa']
    splits = ['train', 'dev', 'test']

    for dataset in datasets:
        for split in splits:
            format_dataset(dataset, split)


if __name__ == '__main__':
    format_all()
