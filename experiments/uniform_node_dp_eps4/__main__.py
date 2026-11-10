"""Normal entry deliberately refuses all real runs; no data-loading arguments."""


def main(argv=None):
    raise PermissionError('REAL_TRAINING_DISABLED: use the reviewed artificial U2 test entry only')


if __name__=='__main__':main()
