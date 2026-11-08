
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model_awal = joblib.load("decision_tree_awal_5fitur.pkl")
model_pruned = joblib.load("decision_tree_pruned_5fitur.pkl")

fitur = ['17', '22', '23', '25', '29']

nama_fitur = {
    '17': 'Jam belajar mingguan',
    '22': 'Kehadiran kelas',
    '23': 'Persiapan ujian pertama',
    '25': 'Kebiasaan mencatat',
    '29': 'IPS semester sebelumnya'
}

keterangan = {
    '17': {
        1: 'Tidak belajar',
        2: 'Kurang dari 5 jam',
        3: '6–10 jam',
        4: '11–20 jam',
        5: 'Lebih dari 20 jam'
    },
    '22': {
        1: 'Selalu hadir',
        2: 'Kadang-kadang hadir',
        3: 'Tidak pernah hadir'
    },
    '23': {
        1: 'Belajar sendiri',
        2: 'Belajar bersama teman',
        3: 'Tidak melakukan persiapan'
    },
    '25': {
        1: 'Tidak pernah mencatat',
        2: 'Kadang-kadang mencatat',
        3: 'Selalu mencatat'
    },
    '29': {
        1: 'Di bawah 2,00',
        2: '2,00–2,49',
        3: '2,50–2,99',
        4: '3,00–3,49',
        5: '3,50–4,00'
    }
}


def ambil_alasan(model, data):
    pohon = model.tree_
    node = 0
    alasan = []
    fitur_diperiksa = []

    while pohon.feature[node] != -2:
        indeks = pohon.feature[node]
        kolom = fitur[indeks]
        nilai = int(data.iloc[0, indeks])
        batas = pohon.threshold[node]

        if nilai <= batas:
            node = pohon.children_left[node]
        else:
            node = pohon.children_right[node]

        nama = nama_fitur[kolom]
        nilai_teks = keterangan[kolom][nilai]

        alasan.append(
            f"{nama} kamu adalah {nilai_teks.lower()}. "
            f"Model menggunakan kondisi ini sebagai salah satu "
            f"pertimbangan dalam menentukan hasil prediksi."
        )

        fitur_diperiksa.append(kolom)

    return alasan, fitur_diperiksa


def buat_saran(hasil, data):
    saran = []

    if hasil == "Tidak Lulus":

        if data['17'] <= 2:
            saran.append(
                "Tingkatkan waktu belajar secara bertahap. "
                "Mulailah dengan membuat jadwal belajar rutin."
            )

        if data['22'] == 3:
            saran.append(
                "Usahakan lebih rutin mengikuti perkuliahan "
                "agar tidak tertinggal materi."
            )
        elif data['22'] == 2:
            saran.append(
                "Tingkatkan konsistensi kehadiran di kelas."
            )

        if data['23'] == 3:
            saran.append(
                "Persiapkan ujian lebih awal dengan mengulang "
                "materi dan mengerjakan latihan soal."
            )

        if data['25'] == 1:
            saran.append(
                "Mulailah mencatat materi penting saat dosen "
                "menjelaskan agar lebih mudah belajar kembali."
            )

        if data['29'] <= 2:
            saran.append(
                "Tingkatkan pemahaman materi dan evaluasi "
                "hasil belajar agar IPS semester berikutnya membaik."
            )

        if not saran:
            saran.append(
                "Pertahankan kebiasaan akademik yang sudah baik. "
                "Evaluasi kembali materi kuliah dan persiapan ujian."
            )

    else:
        saran.append(
            "Pertahankan kebiasaan belajar yang sudah baik."
        )
        saran.append(
            "Tetap jaga kehadiran dan persiapkan ujian dengan rutin."
        )
        saran.append(
            "Terus tingkatkan kemampuan agar prestasimu semakin baik."
        )

    return saran


@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    alasan = []
    saran = []
    model_dipilih = "pruned"

    if request.method == "POST":
        model_dipilih = request.form.get("model", "pruned")

        try:
            data_input = {}

            for nama in fitur:
                nilai = int(request.form[nama])

                if nilai not in keterangan[nama]:
                    raise ValueError("Pilihan tidak valid")

                data_input[nama] = nilai

            data_baru = pd.DataFrame(
                [data_input],
                columns=fitur
            )

            if model_dipilih == "awal":
                model = model_awal
            else:
                model = model_pruned

            hasil = model.predict(data_baru)[0]

            alasan, fitur_diperiksa = ambil_alasan(
                model, data_baru
            )

            saran = buat_saran(hasil, data_input)

        except (ValueError, KeyError):
            hasil = "Input tidak valid. Periksa kembali data."
            alasan = []
            saran = []

    return render_template(
        "index.html",
        hasil=hasil,
        alasan=alasan,
        saran=saran,
        model_dipilih=model_dipilih
    )


if __name__ == "__main__":
    app.run(debug=True)
