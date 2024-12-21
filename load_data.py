import csv
import requests
import os


def download_images(csv_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_url = row['image_url']
            image_id = row['id']

            try:
                response = requests.get(image_url, stream=True)
                response.raise_for_status()

                file_extension = image_url.split('.')[-1]
                file_name = f"{image_id}.{file_extension}"
                file_path = os.path.join(output_folder, file_name)

                with open(file_path, 'wb') as image_file:
                    for chunk in response.iter_content(1024):
                        image_file.write(chunk)

                print(f"Изображение {file_name} успешно сохранено.")

            except requests.exceptions.RequestException as e:
                print(f"Ошибка при загрузке {image_url}: {e}")


csv_file = 'foxes.csv'
output_folder = 'foxes_images'
download_images(csv_file, output_folder)
