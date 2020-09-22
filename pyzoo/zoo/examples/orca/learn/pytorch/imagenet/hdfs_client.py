import argparse
import pyarrow as pa
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        help='The mode for the Spark cluster.')

    args = parser.parse_args()

    import pyarrow as pa

    fs = pa.hdfs.connect()
    with fs.open(args.path, 'rb') as f:
        sys.stdout.buffer.write(f.read(1024))
