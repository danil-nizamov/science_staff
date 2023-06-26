class Validator:

    @staticmethod
    def assert_shapes_equal(*args) -> None:
        for i in range(len(args) - 1):
            assert(
                args[i].shape == args[i + 1].shape,
                "shapes are not equal"
            )
        return
