import csv

def read_euromillions_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            numbers = tuple(map(int, row[:5]))
            data.append(numbers)
    return data

if __name__ == "__main__":
    file_path = 'euromillions.csv'
    euromillions_data = read_euromillions_data(file_path)
    for row in euromillions_data:
        print(row)
