# tampilkan semua nama kolom
print(data_model.columns)

# tampilkan nama kolom yang duplikat
duplikat_kolom = data_model.columns[data_model.columns.duplicated()].unique()
print(duplikat_kolom)