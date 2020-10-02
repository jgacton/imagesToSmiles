import csv
from rdkit import Chem
from rdkit.Chem import Draw
from PIL.PngImagePlugin import PngImageFile, PngInfo

with open("chembl25_smiles.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        elif line_count > 8192:
            break
        else:
            smiles_str = row[0]
            m = Chem.MolFromSmiles(f"{row[0]}")
            Draw.MolToFile(m, f"imagesFromSmilesLite/img{line_count}.png", size=(512, 512))

            image = PngImageFile(f"imagesFromSmilesLite/img{line_count}.png")
            metadata = PngInfo()

            metadata.add_text("SMILES", f"{row[0]}")
            image.save(f"imagesFromSmilesLite/img{line_count}.png", pnginfo=metadata)

            image = PngImageFile(f"imagesFromSmilesLite/img{line_count}.png")

            print(image.text)

            line_count += 1
    print(f'Processed {line_count} lines.')
