Z oryginalnego zbioru walidacyjnego PlotQA wyselekcjonowano pierwsze 30 wykresów typu 'dot_line':
- Dev set ('dev_ids', 10 obrazkow):
- Test set ('test_ids', 20 obrazkow)

Instrukcja odtworzenia zbioru danych:
1. Pobierz i rozpakuj podzbiór walidacyjny PlotQA. Oczekiwana struktura rozpakowanego folderu:
'''text
   dataforsolvroimages/
   ├── annotations/
   │   └── annotations.json
   └── images/
       └── png/
           ├── 0.png
           ├── 1.png
           └── ...
'''

2. Uruchom download_data.py, podajac sciezke do rozpakowanego folderu jako argument:

        python download_data.py "/sciezka/do/zdjec"

