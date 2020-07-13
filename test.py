import argparse

a = 'djjsada'
def tmp(aa):
    print(aa)
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
if  __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments.
    parser.add_argument("--experiment_no", type=int, default=1, required=True, help="using which experiment_no data")
    parser.add_argument("--is_uncased",    type=boolean_string, default=False, required=True, help="convert all upper case into lower case.")

    # Optional arguments.
    parser.add_argument("--batch_size",         type=int,   default=32,     help="determine batch size.")
    parser.add_argument("--dropout",            type=float, default=0,      help="determine dropout.")
    parser.add_argument("--embedding_dim",      type=int,   default=100,    help="determine embedding_dim.")
    parser.add_argument("--epoch",              type=int,   default=10,     help="determine epoch.")
    parser.add_argument("--max_norm",           type=float, default=1,      help="determine max_norm.")
    parser.add_argument("--hidden_dim",         type=int,   default=300,    help="determine hidden_dim.")
    parser.add_argument("--learning_rate",      type=float, default=10e-4,  help="determine learning_rate.")
    parser.add_argument("--min_count",          type=int,   default=0,      help="determine min_count.")
    parser.add_argument("--num_rnn_layers",     type=int,   default=1,      help="determine num_rnn_layers.")
    parser.add_argument("--num_linear_layers",  type=int,   default=2,      help="determine num_linear_layers.")
    parser.add_argument("--seed",               type=int,   default=7,      help="determine seed.")

    args = parser.parse_args()