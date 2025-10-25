# Computer Vision Lab 3 - Task Description

## Część 1: Transformacja Hough

W pierwszej części ćwiczenia należy zapoznać się z algorytmem transformacji Hough.

### Materiały do przestudiowania:

- [Hough Transform - Wikipedia (EN)](https://en.wikipedia.org/wiki/Hough_transform)
- [Transformacja Hougha - Wikipedia (PL)](https://pl.wikipedia.org/wiki/Transformacja_Hougha)
- Praca: "Use of the Hough Transform to Detect Lines and Curves in Pictures", 1972 (zob. folder `./Hough-Line-Detection/HoughTransformPaper.pdf`)

### Zadania do wykonania:

1. **Zapoznanie się ze skryptem** `find_hough_lines.py` do detekcji linii na obrazach.

2. **Przygotowanie notebooka** `find_hough_lines.ipynb` w oparciu o skrypt `find_hough_lines.py`:

   - Zamieścić obrazy ze znalezionymi liniami dla obrazów:
     - `imgs/ex1.png`
     - `imgs/ex2.png`
     - `imgs/ex3.png`
   - Dla różnych parametrów (dyskretyzacji) rho-theta

3. **Implementacja OpenCV**: Przedstawić wyniki uzyskane w oparciu o implementację transformaty Hough z biblioteki OpenCV (cv2).

#### Przykładowe wywołanie OpenCV:

```python
# Use Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply HoughLinesP method
lines_list = []

lines = cv2.HoughLinesP(
    edges,                    # Input edge image
    1,                        # Distance resolution in pixels
    np.pi/180,                # Angle resolution in radians
    threshold=100,            # Min number of votes for valid line
    minLineLength=5,          # Min allowed length of line
    maxLineGap=10             # Max allowed gap between line for joining them
)

# Iterate over detected lines
for points in lines:
    # Extract coordinates from nested list
    x1, y1, x2, y2 = points[0]

    # Draw the lines joining the points on the original image
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Maintain a simple lookup list for points
    lines_list.append([(x1, y1), (x2, y2)])
```

4. **Generowanie raportu**: W oparciu o `find_hough_lines.ipynb` wygenerować `find_hough_lines.pdf`

---

## Część 2: Algorytm RANSAC

W drugiej części ćwiczenia należy zapoznać się z algorytmem RANSAC.

### Materiały do przestudiowania:

- [RANSAC - Wikipedia](https://en.wikipedia.org/wiki/Random_sample_consensus)
- Skrypt `RANSAC_devel.py`
- [RANSAC implementation - scikit-image](https://scikit-image.org/docs/stable/auto_examples/transform/plot_ransac.html)

### Zadania do wykonania:

1. **Przygotowanie notebooka** `RANSAC_devel.ipynb` w oparciu o skrypt `RANSAC_devel.py`
2. **Implementacja scikit-image**: Zamieścić wyniki uzyskane przez RANSAC z biblioteki scikit-image
3. **Generowanie raportu**: W oparciu o `RANSAC_devel.ipynb` wygenerować `RANSAC_devel.pdf`

---

## Część 3: MoveNet - Estymacja pozy człowieka

W trzeciej części ćwiczenia należy wstępnie zapoznać się z siecią MoveNet do estymacji pozy człowieka (zob. folder `moveNet`).

---

## Część 4: Detekcja i anonimizacja twarzy

W czwartej części ćwiczenia należy wstępnie zapoznać się z metodami detekcji twarzy, a w szczególności anonimizacji twarzy (zob. skrypty w folderze `faceBluring`).
