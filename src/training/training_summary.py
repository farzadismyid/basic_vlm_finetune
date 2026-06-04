def print_training_summary(stats):

    print("\nMODEL SUMMARY")
    print("=" * 50)

    print(
        f"Total Parameters: "
        f"{stats['total_parameters']:,}"
    )

    print(
        f"Trainable Parameters: "
        f"{stats['trainable_parameters']:,}"
    )
    