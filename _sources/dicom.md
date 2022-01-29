# DICOMs

For all things involving DICOMs in Python, please use the `pydicom` library. It is the easiest library to use to access, edit, and write DICOM files.

## Retrieving DICOM metadata

First, imagine that your project has this file structure:

```
dicom_example/
├─ main.py
├─ dicom_data/
│  ├─ 1-001.dcm
│  ├─ 1-002.dcm
│  ├─ etc...
```

Now, in order to access the DICOM volume (and metadata), you need to read the a single slice from that volume.

```Python
# Inside main.py
import pydicom

slice_data = pydicom.filereader.dcmread('./dicom_data/1-001.dcm')
```

Now, the `slice_data` variable holds the slice pixel data (as a 2-D array) AND all of its metadata.

To view this metadata, simply print the `slice_data` - it has a custom `__str__()` function (equivalent to Java's `toString()` override) that prints its metadata.

```Python
print(slice_data)
```

Should return something like this:
```
Dataset.file_meta -------------------------------
(0002, 0000) File Meta Information Group Length  UL: 194
(0002, 0001) File Meta Information Version       OB: b'\x00\x01'
(0002, 0002) Media Storage SOP Class UID         UI: CT Image Storage
(0002, 0003) Media Storage SOP Instance UID      UI: 1.3.6.1.4.1.14519.5.2.1.7777.9002.252525588225178607069266994600
(0002, 0010) Transfer Syntax UID                 UI: Implicit VR Little Endian
(0002, 0012) Implementation Class UID            UI: 1.2.40.0.13.1.1.1
...
...
many more lines...
```

From here, you can see all the DICOM metadata.

For most of the metadata tags, you can access by name:
```Python
patient_name = slice_data.PatientName
# Now you have the patient name

orientation = slice_data.ImageOrientation
position = slice_data.ImagePosition
# etc.
```

However, some tags you cannot. 
In those cases, access the variable by 
1. Concatenating the number pairs in the beginning of the metadata line
2. Popping out the leading zero's
3. And throwing them into a hexadecimal number.

Ex:
Say I wanted to access:
`(0002, 0012) Implementation Class UID UI: 1.2.40.0.13.1.1.1`
1. Concatenation: (0002, 0012) -> 00020012
2. Pop out leading zero's -> 20012
3. Convert to hexadecimal -> 0x20012

For example, say I wanted to access `(0028, 1050) Window Center DS: [30, -550]`, something I just see in the metadata print output.

I would access it like this:
```Python
# Tag is (0028, 1050)
# So the key is 0x281050
print(slice_data[0x281050])
```
The output would be:
```
(0028, 1050) Window Center DS: [30, -550]
```

Further information can be read from this [Medium article](https://medium.com/@ashkanpakzad/reading-editing-dicom-metadata-w-python-8204223a59f6). Even better, the official [pydicom documentation](https://pydicom.github.io/pydicom/stable/index.html).

You can also find a barebones example in my [repo](https://github.com/chautrn/DICOM_example).
