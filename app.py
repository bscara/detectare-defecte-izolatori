import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Titlul aplicației
st.title("Detectare automată a defectelor la izolatori")

st.write("Această aplicație folosește un model YOLOv8 antrenat pentru a detecta defecte la izolatori.")

st.info(
    "Modelul poate identifica doar clasele pe care a fost antrenat: "
    "izolator bun, izolator spart, izolator murdar și poluare/flashover."
)

# Încărcarea modelului antrenat
model = YOLO("best.pt")

# Alegerea pragului de încredere
prag = st.slider("Prag de încredere", 0.10, 1.00, 0.45)

# Încărcarea imaginii
imagine_incarcata = st.file_uploader("Încarcă o imagine", type=["jpg", "jpeg", "png"])

if imagine_incarcata is not None:
    imagine = Image.open(imagine_incarcata)

    st.subheader("Imaginea încărcată")
    st.image(imagine, use_container_width=True)

    if st.button("Rulează detecția"):
        # Salvăm temporar imaginea pentru a putea fi analizată de model
        fisier_temporar = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        imagine.save(fisier_temporar.name)

        # Rularea predicției
        rezultate = model.predict(fisier_temporar.name, conf=prag)

        rezultat = rezultate[0]

        # Afișarea imaginii cu detecțiile realizate
        imagine_rezultat = rezultat.plot()

        st.subheader("Rezultatul detecției")
        st.image(imagine_rezultat, use_container_width=True)

        # Afișarea detaliilor
        if len(rezultat.boxes) == 0:
            st.warning("Nu a fost detectat niciun obiect.")
        else:
            st.success(f"Au fost detectate {len(rezultat.boxes)} obiecte.")

            for box in rezultat.boxes:
                id_clasa = int(box.cls[0])
                incredere = float(box.conf[0])
                nume_clasa = model.names[id_clasa]

                if nume_clasa == "Glass_Dirty":
                    nume_clasa = "Izolator murdar"
                elif nume_clasa == "broken":
                    nume_clasa = "Izolator spart"
                elif nume_clasa == "izolator bun":
                    nume_clasa = "Izolator bun"
                elif nume_clasa == "pollution-flashover":
                    nume_clasa = "Poluare / flashover"

                st.write(f"Clasa detectată: **{nume_clasa}**")
                st.write(f"Scor de încredere: **{incredere:.2f}**")
else:
    st.write("Te rog să încarci o imagine pentru testare.")
