import splitfolders

input_folder = "../../data/raw"
output_folder = "../../data/processed"

splitfolders.ratio(
    input_folder,
    output=output_folder,
    seed=70,
    ratio=(0.7, 0.15, 0.15),  # Train:Validation:Test ratio
    group_prefix=None,
    move=False
)
