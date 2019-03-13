import csv


def write_error_log(row, csv_file='bad_pdb.csv'):
    """
    Csv writer function
    :param row:
    :param csv:
    :return:
    """
    with open(csv_file, "a") as csv_file:
        writer = csv.writer((csv_file), delimiter=',')
        if isinstance(row, str):
            row = row.split()
        writer.writerow(row)















