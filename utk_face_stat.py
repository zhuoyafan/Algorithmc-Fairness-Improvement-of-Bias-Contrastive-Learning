import os
import csv

filepath = os.walk("data/utk_face/images")

age = []
gender = []
race = []
for path, dir_list, file_list in filepath:
    for file_name in file_list:
        s = file_name.split("_")
        age.append(s[0])
        gender.append(s[1])
        race.append(s[2])
rows = zip(age, gender, race)
resultpath = "utk_face_stat.csv"
with open(resultpath, "w", newline="") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
