from pathlib import Path


def save_dataframe_to_csv(dataframe, folder_name, file_name):
    output_folder = Path(folder_name)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / file_name

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    return output_path


def save_figure_to_png(figure, folder_name, file_name):
    output_folder = Path(folder_name)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / file_name

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    return output_path