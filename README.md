# plot vision digitizer (PlotQA)

Automatyczny pipeline oparty na **Computer Vision (OpenCV)** oraz regresji liniowej do kalibracji osi i ekstrakcji serii danych $(x, y)$ z jednokolorowych i monochromatycznych wykresów punktowych.

## 📁 Struktura Projektu

```
.
├── data/
│   ├── sample_images/         # Wyselekcjonowany zbiór 30 wykresów (DEV + TEST)
│   └── sample_annotations.json# Wyselekcjonowane metadane i kalibracja
├── notebooks/                 # Notatniki Jupytera z eksperymentami i raportem
│   ├── 01_detector_experiments.ipynb # Eksperymenty i strojenie detektora kropek
│   ├── 02_converter_experiments.ipynb# Eksperymenty z mapowaniem osi i kalibracją
│   └── raport.ipynb                  # Podsumowanie, metryki, sukcesy i porażki
├── src/
│   ├── detector.py            # Moduł detekcji kropek (HSV / Adaptive Thresholding)
│   ├── converter.py           # Mapowanie pikseli na wartości osi X i Y
│   └── main.py                # CLI do uruchamiania pełnego procesu ekstrakcji
├── download_data.py           # Skrypt do selekcji i podziału zbioru 30 wykresów
├── requirements.txt           # Zależności projektu
└── README.md

```

## Wymagania i Instalacja

Projekt wymaga Pythona w wersji **3.10+**.

```
# 1. Tworzenie środowiska wirtualnego
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalacja zależności
pip install opencv-python numpy matplotlib scipy jupyter

```

## Przygotowanie i Selekcja Danych (`download_data.py`)

Skrypt `download_data.py` odpowiada za zbalansowaną selekcję 30 wykresów typu `dot-line` (24 kolorowe + 6 monochromatycznych) z pełnego zbioru PlotQA oraz podział na zbiór deweloperski i testowy.

1. Pobierz i rozpakuj podzbiór walidacyjny PlotQA. Oczekiwana struktura rozpakowanego folderu:

   ```
   your-data/
   ├── annotations/
   │   └── annotations.json
   └── images/
       └── png/
           ├── 0.png
           ├── 1.png
           └── ...
   
   
   ```

2. Uruchom skrypt `download_data.py`, podając ścieżkę do rozpakowanego folderu jako argument:

   ```
   python download_data.py "sciezka do zdjec"
   
   ```

## Szybkie Uruchomienie

Po wyselekcjonowaniu danych możesz uruchomić pełny proces ekstrakcji dla wybranego wykresu:

```
python src/main.py --image data/sample_images/4435.png --config data/sample_annotations.json

```

**Wyniki wyjściowe:**

* `output.json` / `output.csv` – odczytane punkty $(x, y)$.

* `comparison_result.png` – wizualizacja zestawiająca oryginalny obrazek z odtworzonym wykresem (GT vs Pred).

## Notatniki Analityczne (`notebooks/`)

Cały proces badawczy oraz wnioski zostały udokumentowane w katalogu `notebooks/`:

* **`01_detector_experiments.ipynb`** – testowanie algorytmów binaryzacji, filtracji morfologicznej oraz doboru progów kolistości i powierzchni dla wykresów kolorowych i monochromatycznych.

* **`02_converter_experiments.ipynb`** – walidacja algorytmów interpolacji liniowej osi Y oraz przypisywania etykiet kategorii na osi X.

* **`raport.ipynb`** – kompletna ewaluacja (Precision, Recall, F1, MAE), analiza przypadków brzegowych oraz wnioski końcowe.

## Opis Rozwiązania i Architektura

### 1. Detekcja Markerów (`detector.py`)

Detektor automatycznie rozpoznaje charakterystykę obrazu:

* **Wykresy Kolorowe:** Konwersja do przestrzeni HSV, progowanie nasycenia barw oraz filtrowanie konturów pod kątem pola powierzchni oraz okrągłości.

* **Wykresy Monochromatyczne:** Progowanie adaptacyjne, usuwanie linii siatki i osi za pomocą operacji morfologicznych oraz wyznaczanie środków ciężkości momentów konturów.

### 2. Kalibracja i Mapowanie Pikseli (`converter.py`)

* **Oś Y (Wartości Ciągłe):** Wykorzystuje pozycje pikselowe podziałek oraz odpowiadające im wartości numeryczne z konfiguracji. Do przeliczenia pozycji pikselowej na wartość rzeczywistą stosowana jest interpolacja liniowa.

* **Oś X (Kategorie / Dyskretne):** Wyznaczane są środki ramek etykiet/podziałek osi X. Wykryty punkt przypisywany jest do najbliższej etykiety tekstowej na podstawie odległości pikselowej.

### 3. Udział pracy ręcznej vs Automatyzacja

* **Praca ręczna / Konfiguracja:** Dostarczenie struktury osi (położenie ramek osi $X, Y$ oraz wartości znaczników podziałki) pobieranych z metadanych `general_figure_info` (zgodnie z założeniami uproszczenia zakresu).

* **Automatyzacja:** Detekcja punktów, segmentacja markerów od tła/siatki, dopasowanie do najbliższych etykiet i przeliczenie współrzędnych odbywa się w 100% automatycznie.

## Zbiór Danych (PlotQA Subset)

Do analizy i oceny wyselekcjonowano **30 wykresów** typu `dot_line` ze zbioru walidacyjnego PlotQA z zachowaniem stratyfikacji pod kątem kolorystyki:

* **Suma:** 24 wykresy kolorowe + 6 monochromatycznych.

* **Zbiór Deweloperski (DEV - 10 obrazów):** 7 kolorowych, 3 monochromatyczne (użyty do dostrajania parametrów detekcji i debugowania).

* **Zbiór Testowy (TEST - 20 obrazów):** 17 kolorowych, 3 monochromatyczne (użyty do końcowej ewaluacji).

## Metryki Ewaluacji i Wnioski Końcowe

Dopasowanie przewidywanych punktów do referencyjnych realizowane jest przy użyciu **Algorytmu Węgierskiego** z progiem tolerancji równym **0.2% rozpiętości wymiarów obrazu**:

### 1. Główne źródło błędów kalibracyjnych

Analiza przypadków z wykrytymi odchyleniami wskazuje, że drobne błędy $\text{MAE}_Y$ nie wynikają z problemów z samą detekcją kropek, lecz ze sposobu kalibracji osi zaczerpniętego z adnotacji PlotQA. Do wyznaczenia funkcji skalującej użyto środków geometrycznych ($y + h/2$) ramek zawierających etykiety tekstowe na osi. Środek pola tekstowego rzadko pokrywa się idealnie co do piksela z cienką linią podziałki na osi, co wprowadza minimalne, systematyczne przesunięcia wyliczonej funkcji regresji.

### 2. Gdzie rozwiązanie przestaje działać:

* **Obecność linii łączących w wykresach `dot_line`:** Narysowanie ciągłej linii łączącej kropki może fizycznie scalać markery z linią w jeden spójny kontur, przez co filtry geometrii odrzucają obiekt ze względu na spadek kolistości.

* **Nietypowe kształty markerów (np. trójkąty):** Zastosowanie alternatywnych kształtów wskaźników drastycznie obniża współczynnik $\text{circularity}$, co odrzuca je przy sztywno zdefiniowanym progu $0.4$.

* **Nieliniowe osie liczbowe:** Zastosowanie wielomianu pierwszego stopnia (`np.polyfit`) ogranicza model wyłącznie do osi o skalowaniu liniowym.

### 3. Co zmieniłabym w kolejnym podejściu?

* **Przejście na Deep Learning:** Zastąpienie klasycznych filtrów OpenCV dedykowanym modelem detekcji obiektów (np. YOLOv8-pose) w celu uniezależnienia systemu od kolorystyki, tła i kształtu markerów.

* **Autonomiczna detekcja podziałek + OCR:** Wdrożenie modułu OCR połączonego z detekcją kresek podziałki w celu wyeliminowania błędu przesunięcia wynikającego ze środków ramek tekstowych z pliku JSON.

* **Dynamiczne wykrywanie typu skalowania osi:** Rozbudowa modułu `converter.py` o automatyczne rozpoznawanie charakteru osi (liniowa vs logarytmiczna).