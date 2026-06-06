import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from scipy.signal import find_peaks

from src.data_loader import read_signal_file
from src.signal_processing import (
    bandpass_filter,
    calculate_spectrum,
    calculate_rms
)
from src.analysis import analyze_file, recommend_signal_for_analysis
from src.export_utils import save_dataframe_to_csv, save_figure_to_png


st.set_page_config(
    page_title="Аналіз PCG сигналів",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }
    .mode-card {
        padding: 28px;
        border-radius: 18px;
        border: 1px solid #e6e6e6;
        background-color: #fafafa;
        min-height: 190px;
        margin-bottom: 16px;
    }
    .mode-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .mode-text {
        font-size: 16px;
        color: #555;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

SIGNAL_COLORS = {
    "signal_1": "#1f77b4",
    "signal_2": "#ff7f0e",
    "signal_3": "#2ca02c"
}


def save_fragment_pdf_report(
    file_name,
    selected_signal,
    sampling_rate,
    lowcut,
    highcut,
    filter_order,
    min_peak_distance_sec,
    peak_prominence,
    fragment_start,
    fragment_end,
    fragment_rms,
    peaks_count,
    peaks_dataframe,
    analysis_figure
):
    """Створює технічний PDF-звіт для вибраного фрагмента PCG-сигналу."""
    output_folder = Path("reports")
    output_folder.mkdir(parents=True, exist_ok=True)

    safe_file_stem = (
        f"pcg_fragment_report_{fragment_start:.1f}_{fragment_end:.1f}"
        .replace(".", "_")
        .replace(" ", "_")
    )
    safe_file_name = f"{safe_file_stem}.pdf"
    output_path = output_folder / safe_file_name

    with PdfPages(output_path) as pdf:
        summary_fig = plt.figure(figsize=(8.27, 11.69))
        summary_fig.patch.set_facecolor("white")

        summary_fig.text(
            0.5,
            0.94,
            "Звіт розширеного аналізу PCG-сигналу",
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold"
        )

        summary_lines = [
            f"Файл: {file_name}",
            f"Обраний сигнал: {selected_signal}",
            f"Частота дискретизації: {sampling_rate} Гц",
            "",
            "Параметри фільтрації:",
            f"  Нижня межа: {lowcut} Гц",
            f"  Верхня межа: {highcut} Гц",
            f"  Порядок фільтра: {filter_order}",
            "",
            "Параметри пошуку піків:",
            f"  Мінімальна відстань між піками: {min_peak_distance_sec:.2f} с",
            f"  Мінімальна вираженість піка: {peak_prominence:.2f}",
            "",
            "Результати для вибраного фрагмента:",
            f"  Межі фрагмента: {fragment_start:.1f}–{fragment_end:.1f} с",
            f"  RMS відфільтрованого фрагмента: {fragment_rms:.2f}",
            f"  Кількість характерних піків: {peaks_count}",
            "",
            "Короткий висновок:",
            "  У вибраному фрагменті виконано порівняння початкового та",
            "  відфільтрованого PCG-сигналу. На відфільтрованому сигналі",
            "  позначено характерні піки, які можна використовувати для",
            "  подальшого аналізу структури фонокардіографічного запису.",
            "",
            "Примітка: цей звіт не є медичним діагностичним висновком."
        ]

        y_position = 0.86
        for line in summary_lines:
            summary_fig.text(
                0.10,
                y_position,
                line,
                ha="left",
                va="top",
                fontsize=11
            )
            y_position -= 0.035

        pdf.savefig(summary_fig, bbox_inches="tight")
        plt.close(summary_fig)

        pdf.savefig(analysis_figure, bbox_inches="tight")

        table_fig, table_ax = plt.subplots(figsize=(11.69, 8.27))
        table_ax.axis("off")
        table_ax.set_title(
            "Таблиця характерних піків у вибраному фрагменті",
            fontsize=14,
            fontweight="bold",
            pad=20
        )

        if peaks_dataframe.empty:
            table_ax.text(
                0.5,
                0.5,
                "У вибраному фрагменті піки не знайдені.",
                ha="center",
                va="center",
                fontsize=12
            )
        else:
            table_data = peaks_dataframe.copy()
            table_data = table_data.head(20)

            table = table_ax.table(
                cellText=table_data.values,
                colLabels=table_data.columns,
                cellLoc="center",
                loc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.4)

            if len(peaks_dataframe) > 20:
                table_ax.text(
                    0.5,
                    0.08,
                    "У таблиці показано перші 20 піків.",
                    ha="center",
                    va="center",
                    fontsize=10
                )

        pdf.savefig(table_fig, bbox_inches="tight")
        plt.close(table_fig)


    return output_path


def save_full_pdf_report(
    file_name,
    selected_signal,
    sampling_rate,
    lowcut,
    highcut,
    filter_order,
    min_peak_distance_sec,
    peak_prominence,
    dataframe,
    filtered_signal,
    peak_times,
    peak_values,
    peak_prominences,
    peaks_dataframe,
    recommendation_results,
    fragment_start,
    fragment_end,
    fragment_rms,
    fragment_peaks_dataframe,
    advanced_figure
):
    """Створює охайний академічний PDF-звіт з результатами аналізу PCG-сигналу."""
    output_folder = Path("reports")
    output_folder.mkdir(parents=True, exist_ok=True)

    safe_file_stem = (
        f"pcg_report_{fragment_start:.1f}_{fragment_end:.1f}"
        .replace(".", "_")
        .replace(" ", "_")
    )
    output_path = output_folder / f"{safe_file_stem}.pdf"

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    duration = float(dataframe["time_seconds"].max())
    original_signal = dataframe[selected_signal].values
    original_rms = calculate_rms(original_signal)
    filtered_rms = calculate_rms(filtered_signal)

    frequencies, spectrum = calculate_spectrum(filtered_signal, sampling_rate)
    max_frequency_to_report = min(100, int(sampling_rate / 2))
    spectrum_mask = frequencies <= max_frequency_to_report
    visible_frequencies = frequencies[spectrum_mask]
    visible_spectrum = spectrum[spectrum_mask]

    if len(visible_spectrum) > 1:
        dominant_index = np.argmax(visible_spectrum[1:]) + 1
        dominant_frequency = visible_frequencies[dominant_index]
    else:
        dominant_frequency = 0

    full_peaks_count = len(peaks_dataframe)
    fragment_peaks_count = len(fragment_peaks_dataframe)

    if recommendation_results is not None:
        rec_df = pd.DataFrame(recommendation_results)
        best_row = rec_df.loc[rec_df["score"].idxmax()]
        recommendation_text = (
            f"Автоматично рекомендовано {best_row['signal']} "
            f"(оцінка {best_row['score']})."
        )
    else:
        recommendation_text = "Сигнал обрано користувачем вручну."

    with PdfPages(output_path) as pdf:
        # Page 1: academic summary
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        ax.text(
            0.5,
            0.955,
            "Звіт комп’ютерного аналізу PCG-сигналу",
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
            color="#111111"
        )
        ax.text(
            0.5,
            0.925,
            "Підсумкові результати аналізу фонокардіографічного запису",
            ha="center",
            va="top",
            fontsize=10.5,
            color="#555555"
        )

        ax.text(0.08, 0.875, "1. Загальна інформація", fontsize=13, fontweight="bold", color="#111111")

        summary_rows = [
            ["Файл", file_name],
            ["Обраний сигнал", selected_signal],
            ["Кількість рядків", f"{dataframe.shape[0]}"],
            ["Тривалість запису", f"{duration:.2f} с"],
            ["Частота дискретизації", f"{sampling_rate} Гц"],
            ["Діапазон фільтрації", f"{lowcut}-{highcut} Гц"],
            ["Порядок фільтра", f"{filter_order}"],
            ["Параметри пошуку піків", f"min_distance={min_peak_distance_sec:.2f} с; prominence={peak_prominence:.2f}"],
            ["Рекомендація сигналу", recommendation_text],
        ]

        table_ax = fig.add_axes([0.08, 0.59, 0.84, 0.26])
        table_ax.axis("off")
        table = table_ax.table(
            cellText=summary_rows,
            colLabels=["Параметр", "Значення"],
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=[0.36, 0.64]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.45)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#BFC5CC")
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor("#E9EEF5")
                cell.set_text_props(weight="bold", color="#111111")
            elif row % 2 == 0:
                cell.set_facecolor("#F7F9FC")
            else:
                cell.set_facecolor("white")

        ax.text(0.08, 0.535, "2. Основні числові результати", fontsize=13, fontweight="bold", color="#111111")

        result_rows = [
            ["RMS початкового сигналу", f"{original_rms:.2f}"],
            ["RMS після фільтрації", f"{filtered_rms:.2f}"],
            ["Кількість піків у всьому записі", f"{full_peaks_count}"],
            ["Домінантна частота у спектрі", f"{dominant_frequency:.2f} Гц"],
            ["Фрагмент для розширеного аналізу", f"{fragment_start:.1f}-{fragment_end:.1f} с"],
            ["RMS фрагмента", f"{fragment_rms:.2f}"],
            ["Кількість піків у фрагменті", f"{fragment_peaks_count}"],
        ]

        result_ax = fig.add_axes([0.08, 0.32, 0.84, 0.19])
        result_ax.axis("off")
        result_table = result_ax.table(
            cellText=result_rows,
            colLabels=["Показник", "Значення"],
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=[0.56, 0.44]
        )
        result_table.auto_set_font_size(False)
        result_table.set_fontsize(9)
        result_table.scale(1, 1.45)

        for (row, col), cell in result_table.get_celld().items():
            cell.set_edgecolor("#BFC5CC")
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor("#E9EEF5")
                cell.set_text_props(weight="bold", color="#111111")
            elif row % 2 == 0:
                cell.set_facecolor("#F7F9FC")
            else:
                cell.set_facecolor("white")

        ax.text(0.08, 0.255, "3. Короткий висновок", fontsize=13, fontweight="bold", color="#111111")

        conclusion_lines = [
            f"У вибраному фрагменті {fragment_start:.1f}-{fragment_end:.1f} с знайдено {fragment_peaks_count} характерних піків.",
            "Розширений перегляд дозволяє детальніше оцінити форму сигналу після фільтрації",
            "та перевірити роботу алгоритму пошуку піків на окремій ділянці запису.",
            "",
            "Примітка: звіт має навчально-дослідницьке призначення і не є медичним",
            "діагностичним висновком."
        ]

        y = 0.225
        for line in conclusion_lines:
            ax.text(0.08, y, line, fontsize=10, color="#333333", va="top")
            y -= 0.025

        ax.text(
            0.08,
            0.055,
            "Звіт сформовано програмним застосунком для комп’ютерного аналізу PCG-сигналів.",
            fontsize=8.5,
            color="#777777"
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: advanced fragment graph
        pdf.savefig(advanced_figure, bbox_inches="tight")

        # Page 3: peak table and interpretation
        fig_table = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        ax = fig_table.add_axes([0, 0, 1, 1])
        ax.axis("off")

        ax.text(
            0.5,
            0.955,
            "Таблиця характерних піків у вибраному фрагменті",
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
            color="#111111"
        )

        ax.text(
            0.08,
            0.900,
            f"Фрагмент: {fragment_start:.1f}-{fragment_end:.1f} с",
            fontsize=10,
            color="#333333"
        )
        ax.text(
            0.08,
            0.875,
            f"Сигнал: {selected_signal}",
            fontsize=10,
            color="#333333"
        )
        ax.text(
            0.08,
            0.850,
            f"Кількість піків: {fragment_peaks_count}",
            fontsize=10,
            color="#333333"
        )

        if fragment_peaks_dataframe.empty:
            ax.text(
                0.5,
                0.55,
                "У вибраному фрагменті піки не знайдені.",
                ha="center",
                va="center",
                fontsize=12,
                color="#333333"
            )
        else:
            table_data = fragment_peaks_dataframe.copy()
            table_data["Номер піка"] = table_data["Номер піка"].astype(int)
            table_data["Час, с"] = table_data["Час, с"].map(lambda value: f"{value:.4f}")
            table_data["Амплітуда"] = table_data["Амплітуда"].map(lambda value: f"{value:.2f}")
            table_data["Вираженість"] = table_data["Вираженість"].map(lambda value: f"{value:.2f}")

            peak_table_ax = fig_table.add_axes([0.08, 0.48, 0.84, 0.28])
            peak_table_ax.axis("off")

            peak_table = peak_table_ax.table(
                cellText=table_data.values,
                colLabels=table_data.columns,
                cellLoc="center",
                colLoc="center",
                loc="center"
            )
            peak_table.auto_set_font_size(False)
            peak_table.set_fontsize(9)
            peak_table.scale(1, 1.5)

            for (row, col), cell in peak_table.get_celld().items():
                cell.set_edgecolor("#BFC5CC")
                cell.set_linewidth(0.6)
                if row == 0:
                    cell.set_facecolor("#E9EEF5")
                    cell.set_text_props(weight="bold", color="#111111")
                elif row % 2 == 0:
                    cell.set_facecolor("#F7F9FC")
                else:
                    cell.set_facecolor("white")

        ax.text(0.08, 0.365, "Інтерпретація результату", fontsize=13, fontweight="bold", color="#111111")

        interpretation_lines = [
            "Позначені точки є локальними максимумами відфільтрованого сигналу,",
            "які відповідають заданим параметрам пошуку. Вони не трактуються як",
            "медичний діагноз, але можуть бути використані для подальшого дослідження",
            "структури PCG-запису та розвитку алгоритмів сегментації серцевих тонів."
        ]

        y = 0.330
        for line in interpretation_lines:
            ax.text(0.08, y, line, fontsize=10, color="#333333", va="top")
            y -= 0.026

        ax.text(
            0.08,
            0.080,
            "Сформовано програмним застосунком для комп’ютерного аналізу PCG-сигналів.",
            fontsize=8.5,
            color="#777777"
        )

        pdf.savefig(fig_table, bbox_inches="tight")
        plt.close(fig_table)

    return output_path


if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = None


if st.session_state.selected_mode is None:
    st.markdown(
        '<div class="main-title">Комп’ютерний аналіз PCG сигналів</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Оберіть режим роботи програми</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="mode-card">
                <div class="mode-title">Аналіз одного файлу</div>
                <div class="mode-text">
                    Детальний аналіз одного запису: перегляд сигналів, фрагментів,
                    спектра, фільтрації, піків та основних параметрів.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("Почати аналіз одного файлу", use_container_width=True):
            st.session_state.selected_mode = "single"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="mode-card">
                <div class="mode-title">Порівняння кількох файлів</div>
                <div class="mode-text">
                    Аналіз декількох записів одночасно: порівняння RMS,
                    кількості піків та загальної активності сигналів.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("Почати порівняння файлів", use_container_width=True):
            st.session_state.selected_mode = "compare"
            st.rerun()

    st.divider()

    st.info(
        "Спочатку оберіть режим. Після цього відкриються відповідні налаштування "
        "та функції для роботи із PCG-сигналами."
    )

    st.stop()


st.markdown(
    '<div class="main-title">Комп’ютерний аналіз PCG сигналів</div>',
    unsafe_allow_html=True
)

if st.session_state.selected_mode == "single":
    st.markdown(
        '<div class="subtitle">Режим: детальний аналіз одного файлу</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="subtitle">Режим: порівняння кількох файлів</div>',
        unsafe_allow_html=True
    )


st.sidebar.header("Панель керування")

if st.sidebar.button("← Повернутись до вибору режиму", use_container_width=True):
    st.session_state.selected_mode = None
    st.rerun()

st.sidebar.divider()

uploaded_file = None
comparison_files = None

if st.session_state.selected_mode == "single":
    uploaded_file = st.sidebar.file_uploader(
        "Завантажте .txt файл для аналізу",
        type=["txt"],
        key="single_file_upload"
    )

elif st.session_state.selected_mode == "compare":
    comparison_files = st.sidebar.file_uploader(
        "Завантажте .txt файли для порівняння",
        type=["txt"],
        accept_multiple_files=True,
        key="compare_files_upload"
    )

st.sidebar.divider()

sampling_rate = st.sidebar.number_input(
    "Частота дискретизації, Гц",
    min_value=1,
    value=800,
    step=1
)

signal_selection_mode = st.sidebar.radio(
    "Спосіб вибору сигналу",
    ["Автоматично", "Вручну"],
    index=0
)

manual_selected_signal = st.sidebar.selectbox(
    "Оберіть сигнал вручну",
    ["signal_1", "signal_2", "signal_3"],
    index=1,
    disabled=(signal_selection_mode == "Автоматично")
)

selected_signal = "signal_2"

if signal_selection_mode == "Вручну":
    selected_signal = manual_selected_signal

st.sidebar.divider()
st.sidebar.subheader("Параметри фільтрації")

lowcut = st.sidebar.slider(
    "Нижня межа фільтра, Гц",
    min_value=1,
    max_value=100,
    value=20,
    step=1
)

highcut = st.sidebar.slider(
    "Верхня межа фільтра, Гц",
    min_value=lowcut + 1,
    max_value=int(sampling_rate / 2) - 1,
    value=150,
    step=1
)

filter_order = st.sidebar.slider(
    "Порядок фільтра",
    min_value=1,
    max_value=8,
    value=4,
    step=1
)

st.sidebar.divider()
st.sidebar.subheader("Параметри пошуку піків")

min_peak_distance_sec = st.sidebar.slider(
    "Мінімальна відстань між піками, с",
    min_value=0.05,
    max_value=1.50,
    value=0.50,
    step=0.05
)

peak_prominence = st.sidebar.slider(
    "Мінімальна вираженість піка",
    min_value=1.0,
    max_value=200.0,
    value=40.0,
    step=1.0
)


if st.session_state.selected_mode == "single":
    if uploaded_file is not None:
        try:
            df = read_signal_file(uploaded_file, sampling_rate)

            recommended_signal = None
            recommendation_results = None

            if signal_selection_mode == "Автоматично":
                recommended_signal, recommendation_results = recommend_signal_for_analysis(
                    dataframe=df,
                    sampling_rate=sampling_rate,
                    lowcut=lowcut,
                    highcut=highcut,
                    filter_order=filter_order,
                    min_peak_distance_sec=min_peak_distance_sec,
                    peak_prominence=peak_prominence
                )

                selected_signal = recommended_signal

            max_time = float(df["time_seconds"].max())

            st.sidebar.divider()
            st.sidebar.subheader("Фрагмент сигналу")

            start_time = st.sidebar.slider(
                "Початок фрагмента, с",
                min_value=0.0,
                max_value=max_time - 1.0,
                value=0.0,
                step=0.5
            )

            fragment_duration = st.sidebar.slider(
                "Тривалість фрагмента, с",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.5
            )

            end_time = start_time + fragment_duration

            df_fragment = df[
                (df["time_seconds"] >= start_time) &
                (df["time_seconds"] <= end_time)
            ]

            signal_data = df[selected_signal].values

            filtered_signal = bandpass_filter(
                signal=signal_data,
                sampling_rate=sampling_rate,
                lowcut=lowcut,
                highcut=highcut,
                order=filter_order
            )

            df["filtered_signal"] = filtered_signal

            df_filtered_fragment = df[
                (df["time_seconds"] >= start_time) &
                (df["time_seconds"] <= end_time)
            ]

            min_peak_distance_samples = max(
                1,
                int(min_peak_distance_sec * sampling_rate)
            )

            peaks, peak_properties = find_peaks(
                filtered_signal,
                distance=min_peak_distance_samples,
                prominence=peak_prominence
            )

            peak_times = df["time_seconds"].iloc[peaks].values
            peak_values = filtered_signal[peaks]
            peak_prominences = peak_properties["prominences"]

            peaks_df = pd.DataFrame({
                "Номер піка": np.arange(1, len(peaks) + 1),
                "Час, с": np.round(peak_times, 4),
                "Амплітуда": np.round(peak_values, 4),
                "Вираженість": np.round(peak_prominences, 4)
            })

            fragment_peaks_mask = (
                (peak_times >= start_time) &
                (peak_times <= end_time)
            )

            fragment_peak_times = peak_times[fragment_peaks_mask]
            fragment_peak_values = peak_values[fragment_peaks_mask]

            st.success(f"Файл успішно завантажено: {uploaded_file.name}")

            if signal_selection_mode == "Автоматично":
                st.info(
                    f"Автоматично рекомендований сигнал для аналізу: `{selected_signal}`. "
                    "Рекомендація сформована на основі характеристик сигналу після фільтрації."
                )
            else:
                st.info(
                    f"Для аналізу вручну обрано сигнал: `{selected_signal}`."
                )

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                st.metric("Кількість рядків", df.shape[0])

            with metric_col2:
                st.metric("Кількість колонок", 4)

            with metric_col3:
                st.metric("Частота", f"{sampling_rate} Гц")

            with metric_col4:
                st.metric("Тривалість", f"{max_time:.2f} с")

            st.divider()

            tab_overview, tab_signal, tab_compare, tab_fragment, tab_advanced, tab_spectrum, tab_filter, tab_peaks, tab_stats = st.tabs(
                [
                    "Огляд",
                    "Графіки",
                    "Порівняння сигналів",
                    "Фрагмент",
                    "Розширений аналіз",
                    "Спектр",
                    "Фільтрація",
                    "Піки",
                    "Параметри"
                ]
            )

            with tab_overview:
                st.subheader("Огляд завантажених даних")

                st.write(
                    "Файл зчитується як таблиця з чотирма початковими колонками. "
                    "Перша колонка використовується як часовий індекс, інші три — як сигнали."
                )

                with st.expander("Показати перші рядки таблиці", expanded=True):
                    st.dataframe(df.head(), use_container_width=True)

                with st.expander("Показати службову інформацію"):
                    st.write(f"Назва файлу: `{uploaded_file.name}`")
                    st.write(f"Кількість рядків: `{df.shape[0]}`")
                    st.write("Кількість початкових колонок: `4`")
                    st.write(f"Частота дискретизації: `{sampling_rate} Гц`")
                    st.write(f"Тривалість сигналу: `{max_time:.2f} секунд`")

                if recommendation_results is not None:
                    with st.expander("Показати результати автоматичної рекомендації сигналу"):
                        recommendation_df = pd.DataFrame(recommendation_results)
                        st.dataframe(recommendation_df, use_container_width=True)

            with tab_signal:
                st.subheader("Візуалізація сигналів")

                graph_col1, graph_col2 = st.columns(2)

                with graph_col1:
                    st.markdown("#### Усі сигнали")

                    fig_all, ax_all = plt.subplots(figsize=(8, 4))

                    for signal_name in ["signal_1", "signal_2", "signal_3"]:
                        ax_all.plot(
                            df["time_seconds"],
                            df[signal_name],
                            label=signal_name,
                            color=SIGNAL_COLORS[signal_name]
                        )

                    ax_all.set_title("Усі сигнали")
                    ax_all.set_xlabel("Час, с")
                    ax_all.set_ylabel("Амплітуда")
                    ax_all.legend()
                    ax_all.grid(True)

                    st.pyplot(fig_all)

                with graph_col2:
                    st.markdown(f"#### Обраний сигнал: `{selected_signal}`")

                    fig_one, ax_one = plt.subplots(figsize=(8, 4))

                    ax_one.plot(
                        df["time_seconds"],
                        df[selected_signal],
                        label=selected_signal,
                        color=SIGNAL_COLORS[selected_signal]
                    )

                    ax_one.set_title(f"Окремий графік: {selected_signal}")
                    ax_one.set_xlabel("Час, с")
                    ax_one.set_ylabel("Амплітуда")
                    ax_one.legend()
                    ax_one.grid(True)

                    st.pyplot(fig_one)

            with tab_compare:
                st.subheader("Порівняння коротких фрагментів сигналів")

                st.info(
                    f"Порівнюється фрагмент від {start_time:.1f} с до {end_time:.1f} с."
                )

                compare_col1, compare_col2, compare_col3 = st.columns(3)

                with compare_col1:
                    st.markdown("#### signal_1")

                    fig_s1, ax_s1 = plt.subplots(figsize=(5, 3))
                    ax_s1.plot(
                        df_fragment["time_seconds"],
                        df_fragment["signal_1"],
                        color=SIGNAL_COLORS["signal_1"]
                    )
                    ax_s1.set_title("signal_1")
                    ax_s1.set_xlabel("Час, с")
                    ax_s1.set_ylabel("Амплітуда")
                    ax_s1.grid(True)
                    st.pyplot(fig_s1)

                with compare_col2:
                    st.markdown("#### signal_2")

                    fig_s2, ax_s2 = plt.subplots(figsize=(5, 3))
                    ax_s2.plot(
                        df_fragment["time_seconds"],
                        df_fragment["signal_2"],
                        color=SIGNAL_COLORS["signal_2"]
                    )
                    ax_s2.set_title("signal_2")
                    ax_s2.set_xlabel("Час, с")
                    ax_s2.set_ylabel("Амплітуда")
                    ax_s2.grid(True)
                    st.pyplot(fig_s2)

                with compare_col3:
                    st.markdown("#### signal_3")

                    fig_s3, ax_s3 = plt.subplots(figsize=(5, 3))
                    ax_s3.plot(
                        df_fragment["time_seconds"],
                        df_fragment["signal_3"],
                        color=SIGNAL_COLORS["signal_3"]
                    )
                    ax_s3.set_title("signal_3")
                    ax_s3.set_xlabel("Час, с")
                    ax_s3.set_ylabel("Амплітуда")
                    ax_s3.grid(True)
                    st.pyplot(fig_s3)

                if signal_selection_mode == "Автоматично":
                    st.warning(
                        f"На цьому етапі програма автоматично рекомендує `{selected_signal}` "
                        "як найбільш придатний сигнал для подальшого аналізу. "
                        "Рекомендація є попередньою і може бути змінена користувачем у ручному режимі."
                    )
                else:
                    st.warning(
                        f"Для подальшого аналізу користувач вручну обрав `{selected_signal}`."
                    )

            with tab_fragment:
                st.subheader("Детальний перегляд фрагмента сигналу")

                st.info(
                    f"Показано фрагмент сигналу `{selected_signal}` "
                    f"від {start_time:.1f} с до {end_time:.1f} с."
                )

                fig_fragment, ax_fragment = plt.subplots(figsize=(14, 5))

                ax_fragment.plot(
                    df_fragment["time_seconds"],
                    df_fragment[selected_signal],
                    label=f"{selected_signal} ({start_time:.1f}–{end_time:.1f} с)",
                    color=SIGNAL_COLORS[selected_signal]
                )

                ax_fragment.set_title(f"Фрагмент сигналу: {selected_signal}")
                ax_fragment.set_xlabel("Час, с")
                ax_fragment.set_ylabel("Амплітуда")
                ax_fragment.legend()
                ax_fragment.grid(True)

                st.pyplot(fig_fragment)

            with tab_advanced:
                st.subheader("Розширений аналіз фрагмента PCG-сигналу")

                st.write(
                    "У цьому блоці можна вручну вибрати короткий фрагмент запису, "
                    "переглянути початковий та відфільтрований сигнал, а також оцінити "
                    "знайдені характерні піки саме на цій ділянці."
                )

                st.info(
                    "Для основного тестового файлу найзручнішим демонстраційним фрагментом "
                    "є ділянка 5.0–10.0 с, але межі можна змінити вручну."
                )

                if max_time <= 1.0:
                    st.warning("Запис надто короткий для розширеного аналізу фрагмента.")
                else:
                    # За замовчуванням обрано найбільш наочний фрагмент для основного тестового файлу.
                    # Користувач може змінити межі фрагмента вручну.
                    adv_default_start = 5.0 if max_time >= 10.0 else (1.0 if max_time >= 6.0 else 0.0)
                    adv_default_end = min(adv_default_start + 5.0, max_time)

                    range_col1, range_col2, range_col3 = st.columns([1, 1, 1])

                    with range_col1:
                        advanced_start_time = st.number_input(
                            "Початок фрагмента, с",
                            min_value=0.0,
                            max_value=max(0.0, max_time - 0.1),
                            value=float(adv_default_start),
                            step=0.1,
                            format="%.1f",
                            key="advanced_start_time"
                        )

                    with range_col2:
                        advanced_end_time = st.number_input(
                            "Кінець фрагмента, с",
                            min_value=min(max_time, advanced_start_time + 0.1),
                            max_value=max_time,
                            value=float(max(adv_default_end, advanced_start_time + 0.1)),
                            step=0.1,
                            format="%.1f",
                            key="advanced_end_time"
                        )

                    with range_col3:
                        show_peak_numbers = st.checkbox(
                            "Показувати номери піків",
                            value=False,
                            key="show_advanced_peak_numbers"
                        )

                    if advanced_end_time <= advanced_start_time:
                        st.warning("Кінець фрагмента має бути більшим за початок.")
                    else:
                        advanced_fragment_df = df[
                            (df["time_seconds"] >= advanced_start_time) &
                            (df["time_seconds"] <= advanced_end_time)
                        ].copy()

                        advanced_peaks_mask = (
                            (peak_times >= advanced_start_time) &
                            (peak_times <= advanced_end_time)
                        )

                        advanced_peak_times = peak_times[advanced_peaks_mask]
                        advanced_peak_values = peak_values[advanced_peaks_mask]
                        advanced_peak_prominences = peak_prominences[advanced_peaks_mask]

                        advanced_peaks_df = pd.DataFrame({
                            "Номер піка": np.arange(1, len(advanced_peak_times) + 1),
                            "Час, с": np.round(advanced_peak_times, 4),
                            "Амплітуда": np.round(advanced_peak_values, 4),
                            "Вираженість": np.round(advanced_peak_prominences, 4)
                        })

                        advanced_fragment_rms = calculate_rms(
                            advanced_fragment_df["filtered_signal"].values
                        )

                        advanced_duration = advanced_end_time - advanced_start_time

                        adv_metric_col1, adv_metric_col2, adv_metric_col3, adv_metric_col4 = st.columns(4)

                        with adv_metric_col1:
                            st.metric("Початок фрагмента", f"{advanced_start_time:.1f} с")

                        with adv_metric_col2:
                            st.metric("Кінець фрагмента", f"{advanced_end_time:.1f} с")

                        with adv_metric_col3:
                            st.metric("Піків у фрагменті", len(advanced_peak_times))

                        with adv_metric_col4:
                            st.metric("RMS фрагмента", f"{advanced_fragment_rms:.2f}")

                        fig_advanced, (ax_raw_fragment, ax_filtered_fragment) = plt.subplots(
                            2,
                            1,
                            figsize=(14, 8),
                            sharex=True
                        )

                        ax_raw_fragment.plot(
                            advanced_fragment_df["time_seconds"],
                            advanced_fragment_df[selected_signal],
                            label="Початковий сигнал",
                            color=SIGNAL_COLORS[selected_signal],
                            linewidth=1.2,
                            alpha=0.75
                        )
                        ax_raw_fragment.set_title(f"Початковий фрагмент сигналу: {selected_signal}")
                        ax_raw_fragment.set_ylabel("Амплітуда")
                        ax_raw_fragment.legend(loc="upper right")
                        ax_raw_fragment.grid(True, alpha=0.3)

                        ax_filtered_fragment.plot(
                            advanced_fragment_df["time_seconds"],
                            advanced_fragment_df["filtered_signal"],
                            label="Відфільтрований сигнал",
                            color="#d62728",
                            linewidth=1.2
                        )

                        ax_filtered_fragment.scatter(
                            advanced_peak_times,
                            advanced_peak_values,
                            label="Характерні піки",
                            marker="o",
                            s=36,
                            color=SIGNAL_COLORS[selected_signal],
                            alpha=0.95,
                            zorder=3
                        )

                        if show_peak_numbers:
                            for peak_number, peak_time, peak_value in zip(
                                advanced_peaks_df["Номер піка"],
                                advanced_peak_times,
                                advanced_peak_values
                            ):
                                ax_filtered_fragment.annotate(
                                    str(peak_number),
                                    (peak_time, peak_value),
                                    textcoords="offset points",
                                    xytext=(0, 8),
                                    ha="center",
                                    fontsize=8
                                )

                        ax_filtered_fragment.set_title(
                            "Відфільтрований фрагмент з характерними піками"
                        )
                        ax_filtered_fragment.set_xlabel("Час, с")
                        ax_filtered_fragment.set_ylabel("Амплітуда")
                        ax_filtered_fragment.legend(loc="upper right")
                        ax_filtered_fragment.grid(True, alpha=0.3)

                        fig_advanced.suptitle(
                            f"Розширений аналіз фрагмента відфільтрованого PCG-сигналу "
                            f"({advanced_start_time:.1f}–{advanced_end_time:.1f} с)",
                            fontsize=14,
                            fontweight="bold"
                        )
                        fig_advanced.tight_layout()

                        st.pyplot(fig_advanced)

                        st.success(
                            f"У вибраному фрагменті тривалістю {advanced_duration:.1f} с "
                            f"знайдено {len(advanced_peak_times)} характерних піків. "
                            f"RMS відфільтрованого фрагмента становить {advanced_fragment_rms:.2f}."
                        )

                        with st.expander("Показати таблицю піків у вибраному фрагменті", expanded=True):
                            if advanced_peaks_df.empty:
                                st.info("У вибраному фрагменті піки не знайдені. Спробуйте змінити межі фрагмента або параметри пошуку піків.")
                            else:
                                st.dataframe(advanced_peaks_df, use_container_width=True)

                        save_col1, save_col2, save_col3 = st.columns(3)

                        with save_col1:
                            if st.button(
                                "Зберегти графік розширеного аналізу у PNG",
                                use_container_width=True,
                                key="save_advanced_plot_png"
                            ):
                                saved_path = save_figure_to_png(
                                    figure=fig_advanced,
                                    folder_name="figures",
                                    file_name=f"advanced_fragment_analysis_{advanced_start_time:.1f}_{advanced_end_time:.1f}.png".replace(".", "_")
                                )

                                st.success(f"Графік збережено: {saved_path}")

                        with save_col2:
                            if st.button(
                                "Зберегти таблицю піків фрагмента у CSV",
                                use_container_width=True,
                                key="save_advanced_peaks_csv"
                            ):
                                saved_path = save_dataframe_to_csv(
                                    dataframe=advanced_peaks_df,
                                    folder_name="results",
                                    file_name=f"advanced_fragment_peaks_{advanced_start_time:.1f}_{advanced_end_time:.1f}.csv".replace(".", "_")
                                )

                                st.success(f"Таблицю збережено: {saved_path}")

                        with save_col3:
                            if st.button(
                                "Сформувати повний PDF-звіт",
                                use_container_width=True,
                                key="save_full_pdf_report"
                            ):
                                saved_path = save_full_pdf_report(
                                    file_name=uploaded_file.name,
                                    selected_signal=selected_signal,
                                    sampling_rate=sampling_rate,
                                    lowcut=lowcut,
                                    highcut=highcut,
                                    filter_order=filter_order,
                                    min_peak_distance_sec=min_peak_distance_sec,
                                    peak_prominence=peak_prominence,
                                    dataframe=df,
                                    filtered_signal=filtered_signal,
                                    peak_times=peak_times,
                                    peak_values=peak_values,
                                    peak_prominences=peak_prominences,
                                    peaks_dataframe=peaks_df,
                                    recommendation_results=recommendation_results,
                                    fragment_start=advanced_start_time,
                                    fragment_end=advanced_end_time,
                                    fragment_rms=advanced_fragment_rms,
                                    fragment_peaks_dataframe=advanced_peaks_df,
                                    advanced_figure=fig_advanced
                                )

                                st.session_state["full_pdf_path"] = str(saved_path)
                                st.success(f"Повний PDF-звіт сформовано: {saved_path}")

                            if "full_pdf_path" in st.session_state:
                                pdf_path = Path(st.session_state["full_pdf_path"])

                                if pdf_path.exists():
                                    with open(pdf_path, "rb") as pdf_file:
                                        st.download_button(
                                            label="Завантажити повний PDF-звіт",
                                            data=pdf_file,
                                            file_name=pdf_path.name,
                                            mime="application/pdf",
                                            use_container_width=True,
                                            key="download_full_pdf_report"
                                        )

                        st.caption(
                            "Цей блок не виконує медичну сегментацію S1/S2, а показує "
                            "характерні піки у вибраному фрагменті після фільтрації. "
                            "Такий результат можна використовувати як основу для подальшого розвитку алгоритму."
                        )

            with tab_spectrum:
                st.subheader("Спектральний аналіз сигналів")

                max_frequency_to_show = st.slider(
                    "Максимальна частота для відображення, Гц",
                    min_value=10,
                    max_value=int(sampling_rate / 2),
                    value=100,
                    step=10
                )

                st.markdown(f"#### Спектр вибраного сигналу: `{selected_signal}`")

                frequencies, spectrum = calculate_spectrum(
                    signal_data,
                    sampling_rate
                )

                spectrum_mask = frequencies <= max_frequency_to_show

                fig_spectrum, ax_spectrum = plt.subplots(figsize=(14, 5))

                ax_spectrum.plot(
                    frequencies[spectrum_mask],
                    spectrum[spectrum_mask],
                    color=SIGNAL_COLORS[selected_signal]
                )

                ax_spectrum.set_title(f"Спектр сигналу: {selected_signal}")
                ax_spectrum.set_xlabel("Частота, Гц")
                ax_spectrum.set_ylabel("Амплітуда спектра")
                ax_spectrum.grid(True)

                st.pyplot(fig_spectrum)

                visible_frequencies = frequencies[spectrum_mask]
                visible_spectrum = spectrum[spectrum_mask]

                if len(visible_spectrum) > 1:
                    dominant_index_visible = np.argmax(visible_spectrum[1:]) + 1
                    dominant_frequency_visible = visible_frequencies[dominant_index_visible]
                    dominant_amplitude_visible = visible_spectrum[dominant_index_visible]
                else:
                    dominant_frequency_visible = 0
                    dominant_amplitude_visible = 0

                spec_col1, spec_col2 = st.columns(2)

                with spec_col1:
                    st.metric(
                        "Домінантна частота в показаному діапазоні",
                        f"{dominant_frequency_visible:.2f} Гц"
                    )

                with spec_col2:
                    st.metric(
                        "Амплітуда на цій частоті",
                        f"{dominant_amplitude_visible:.2f}"
                    )

                st.info(
                    f"Для сигналу `{selected_signal}` найбільша амплітуда в межах показаного діапазону "
                    f"спостерігається приблизно на частоті {dominant_frequency_visible:.2f} Гц. "
                    f"Оскільки частота дискретизації становить {sampling_rate} Гц, "
                    f"максимальна частота коректного спектрального аналізу дорівнює "
                    f"{sampling_rate / 2:.0f} Гц."
                )

                st.divider()

                st.markdown("#### Порівняння спектрів трьох сигналів")

                fig_compare_spectrum, ax_compare_spectrum = plt.subplots(figsize=(14, 5))

                for signal_name in ["signal_1", "signal_2", "signal_3"]:
                    current_signal = df[signal_name].values

                    current_frequencies, current_spectrum = calculate_spectrum(
                        current_signal,
                        sampling_rate
                    )

                    ax_compare_spectrum.plot(
                        current_frequencies[spectrum_mask],
                        current_spectrum[spectrum_mask],
                        label=signal_name,
                        color=SIGNAL_COLORS[signal_name]
                    )

                ax_compare_spectrum.set_title("Порівняння спектрів signal_1, signal_2, signal_3")
                ax_compare_spectrum.set_xlabel("Частота, Гц")
                ax_compare_spectrum.set_ylabel("Амплітуда спектра")
                ax_compare_spectrum.legend()
                ax_compare_spectrum.grid(True)

                st.pyplot(fig_compare_spectrum)

            with tab_filter:
                st.subheader("Фільтрація вибраного сигналу")

                st.info(
                    f"Для сигналу `{selected_signal}` застосовано смуговий фільтр "
                    f"від {lowcut} Гц до {highcut} Гц, порядок фільтра: {filter_order}."
                )

                st.markdown("#### Порівняння сигналу до та після фільтрації")

                fig_filter_full, ax_filter_full = plt.subplots(figsize=(14, 5))

                ax_filter_full.plot(
                    df["time_seconds"],
                    df[selected_signal],
                    label="Початковий сигнал",
                    alpha=0.6,
                    color=SIGNAL_COLORS[selected_signal]
                )

                ax_filter_full.plot(
                    df["time_seconds"],
                    df["filtered_signal"],
                    label="Відфільтрований сигнал",
                    alpha=0.95,
                    color="#d62728"
                )

                ax_filter_full.set_title(f"Фільтрація сигналу: {selected_signal}")
                ax_filter_full.set_xlabel("Час, с")
                ax_filter_full.set_ylabel("Амплітуда")
                ax_filter_full.legend()
                ax_filter_full.grid(True)

                st.pyplot(fig_filter_full)

                st.markdown("#### Фрагмент сигналу до та після фільтрації")

                fig_filter_fragment, ax_filter_fragment = plt.subplots(figsize=(14, 5))

                ax_filter_fragment.plot(
                    df_filtered_fragment["time_seconds"],
                    df_filtered_fragment[selected_signal],
                    label="Початковий фрагмент",
                    alpha=0.6,
                    color=SIGNAL_COLORS[selected_signal]
                )

                ax_filter_fragment.plot(
                    df_filtered_fragment["time_seconds"],
                    df_filtered_fragment["filtered_signal"],
                    label="Відфільтрований фрагмент",
                    alpha=0.95,
                    color="#d62728"
                )

                ax_filter_fragment.set_title(
                    f"Фрагмент після фільтрації: {selected_signal} "
                    f"({start_time:.1f}–{end_time:.1f} с)"
                )
                ax_filter_fragment.set_xlabel("Час, с")
                ax_filter_fragment.set_ylabel("Амплітуда")
                ax_filter_fragment.legend()
                ax_filter_fragment.grid(True)

                st.pyplot(fig_filter_fragment)

            with tab_peaks:
                st.subheader("Пошук піків у відфільтрованому сигналі")

                peaks_col1, peaks_col2, peaks_col3 = st.columns(3)

                with peaks_col1:
                    st.metric("Кількість знайдених піків", len(peaks))

                with peaks_col2:
                    st.metric(
                        "Мін. відстань між піками",
                        f"{min_peak_distance_sec:.2f} с"
                    )

                with peaks_col3:
                    st.metric(
                        "Мін. вираженість піка",
                        f"{peak_prominence:.2f}"
                    )

                st.info(
                    f"У відфільтрованому сигналі `{selected_signal}` знайдено {len(peaks)} піків. "
                    "Ці піки використовуються як інформативні точки для подальшого аналізу PCG."
                )

                st.markdown("#### Піки на повному відфільтрованому сигналі")

                fig_peaks_full, ax_peaks_full = plt.subplots(figsize=(14, 5))

                ax_peaks_full.plot(
                    df["time_seconds"],
                    df["filtered_signal"],
                    label="Відфільтрований сигнал",
                    color="#d62728"
                )

                ax_peaks_full.scatter(
                    peak_times,
                    peak_values,
                    label="Знайдені піки",
                    marker="o",
                    color=SIGNAL_COLORS[selected_signal]
                )

                ax_peaks_full.set_title(f"Знайдені піки: {selected_signal}")
                ax_peaks_full.set_xlabel("Час, с")
                ax_peaks_full.set_ylabel("Амплітуда")
                ax_peaks_full.legend()
                ax_peaks_full.grid(True)

                st.pyplot(fig_peaks_full)

                if st.button("Зберегти графік піків у figures", use_container_width=True):
                    saved_path = save_figure_to_png(
                        figure=fig_peaks_full,
                        folder_name="figures",
                        file_name="peaks_plot.png"
                    )

                    st.success(f"Графік піків збережено: {saved_path}")

                st.markdown("#### Піки на вибраному фрагменті")

                fig_peaks_fragment, ax_peaks_fragment = plt.subplots(figsize=(14, 5))

                ax_peaks_fragment.plot(
                    df_filtered_fragment["time_seconds"],
                    df_filtered_fragment["filtered_signal"],
                    label="Відфільтрований фрагмент",
                    color="#d62728"
                )

                ax_peaks_fragment.scatter(
                    fragment_peak_times,
                    fragment_peak_values,
                    label="Піки у фрагменті",
                    marker="o",
                    color=SIGNAL_COLORS[selected_signal]
                )

                ax_peaks_fragment.set_title(
                    f"Піки на фрагменті: {selected_signal} "
                    f"({start_time:.1f}–{end_time:.1f} с)"
                )
                ax_peaks_fragment.set_xlabel("Час, с")
                ax_peaks_fragment.set_ylabel("Амплітуда")
                ax_peaks_fragment.legend()
                ax_peaks_fragment.grid(True)

                st.pyplot(fig_peaks_fragment)

                with st.expander("Показати таблицю знайдених піків"):
                    st.dataframe(peaks_df, use_container_width=True)

                if st.button("Зберегти таблицю піків у results", use_container_width=True):
                    saved_path = save_dataframe_to_csv(
                        dataframe=peaks_df,
                        folder_name="results",
                        file_name="peaks_table.csv"
                    )

                    st.success(f"Таблицю піків збережено: {saved_path}")

            with tab_stats:
                st.subheader("Базові характеристики вибраного сигналу")

                min_value = signal_data.min()
                max_value = signal_data.max()
                mean_value = signal_data.mean()
                std_value = signal_data.std()
                rms_value = calculate_rms(signal_data)

                filtered_min = filtered_signal.min()
                filtered_max = filtered_signal.max()
                filtered_mean = filtered_signal.mean()
                filtered_std = filtered_signal.std()
                filtered_rms = calculate_rms(filtered_signal)

                stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)

                with stats_col1:
                    st.metric("Мінімум", f"{min_value:.2f}")

                with stats_col2:
                    st.metric("Максимум", f"{max_value:.2f}")

                with stats_col3:
                    st.metric("Середнє", f"{mean_value:.2f}")

                with stats_col4:
                    st.metric("Станд. відхилення", f"{std_value:.2f}")

                with stats_col5:
                    st.metric("RMS", f"{rms_value:.2f}")

                characteristics_df = pd.DataFrame({
                    "Параметр": [
                        "Мінімальне значення",
                        "Максимальне значення",
                        "Середнє значення",
                        "Стандартне відхилення",
                        "RMS",
                        "Кількість піків"
                    ],
                    "Початковий сигнал": [
                        round(min_value, 4),
                        round(max_value, 4),
                        round(mean_value, 4),
                        round(std_value, 4),
                        round(rms_value, 4),
                        "-"
                    ],
                    "Відфільтрований сигнал": [
                        round(filtered_min, 4),
                        round(filtered_max, 4),
                        round(filtered_mean, 4),
                        round(filtered_std, 4),
                        round(filtered_rms, 4),
                        len(peaks)
                    ]
                })

                with st.expander("Показати таблицю характеристик", expanded=True):
                    st.dataframe(characteristics_df, use_container_width=True)

        except Exception as e:
            st.error("Не вдалося зчитати файл або виконати обробку.")
            st.write(e)

    else:
        st.warning("Файл для аналізу ще не завантажено.")

        st.info(
            "У цьому режимі потрібно завантажити один .txt файл. "
            "Після цього програма покаже графіки, спектр, фільтрацію, піки та характеристики сигналу."
        )


if st.session_state.selected_mode == "compare":
    if comparison_files:
        try:
            comparison_results = []

            for file in comparison_files:
                result = analyze_file(
                    file=file,
                    sampling_rate=sampling_rate,
                    selected_signal=selected_signal,
                    lowcut=lowcut,
                    highcut=highcut,
                    filter_order=filter_order,
                    min_peak_distance_sec=min_peak_distance_sec,
                    peak_prominence=peak_prominence
                )

                comparison_results.append(result)

            comparison_df = pd.DataFrame(comparison_results)

            st.success("Порівняння файлів виконано.")

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Кількість файлів", len(comparison_files))

            with metric_col2:
                max_peaks_file = comparison_df.loc[
                    comparison_df["Кількість піків"].idxmax(),
                    "Файл"
                ]
                st.metric("Найбільше піків", max_peaks_file)

            with metric_col3:
                max_rms_file = comparison_df.loc[
                    comparison_df["RMS після фільтрації"].idxmax(),
                    "Файл"
                ]
                st.metric("Найвище RMS після фільтрації", max_rms_file)

            st.divider()

            st.markdown("#### Таблиця порівняння файлів")
            st.dataframe(comparison_df, use_container_width=True)

            if st.button("Зберегти таблицю порівняння у results", use_container_width=True):
                saved_path = save_dataframe_to_csv(
                    dataframe=comparison_df,
                    folder_name="results",
                    file_name="comparison_results.csv"
                )

                st.success(f"Таблицю порівняння збережено: {saved_path}")

            st.markdown("#### Кількість знайдених піків у кожному файлі")

            fig_files_peaks, ax_files_peaks = plt.subplots(figsize=(12, 5))

            ax_files_peaks.bar(
                comparison_df["Файл"],
                comparison_df["Кількість піків"],
                color="#1f77b4"
            )

            ax_files_peaks.set_title("Порівняння кількості піків у файлах")
            ax_files_peaks.set_xlabel("Файл")
            ax_files_peaks.set_ylabel("Кількість піків")
            ax_files_peaks.tick_params(axis="x", rotation=25)
            ax_files_peaks.grid(True, axis="y")

            for index, value in enumerate(comparison_df["Кількість піків"]):
                ax_files_peaks.text(
                    index,
                    value,
                    str(value),
                    ha="center",
                    va="bottom"
                )

            st.pyplot(fig_files_peaks)

            if st.button("Зберегти графік кількості піків у figures", use_container_width=True):
                saved_path = save_figure_to_png(
                    figure=fig_files_peaks,
                    folder_name="figures",
                    file_name="comparison_peaks.png"
                )

                st.success(f"Графік кількості піків збережено: {saved_path}")

            st.markdown("#### RMS після фільтрації для кожного файлу")

            fig_files_rms, ax_files_rms = plt.subplots(figsize=(12, 5))

            ax_files_rms.bar(
                comparison_df["Файл"],
                comparison_df["RMS після фільтрації"],
                color="#1f77b4"
            )

            ax_files_rms.set_title("Порівняння RMS після фільтрації")
            ax_files_rms.set_xlabel("Файл")
            ax_files_rms.set_ylabel("RMS")
            ax_files_rms.tick_params(axis="x", rotation=25)
            ax_files_rms.grid(True, axis="y")

            for index, value in enumerate(comparison_df["RMS після фільтрації"]):
                ax_files_rms.text(
                    index,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom"
                )

            st.pyplot(fig_files_rms)

            if st.button("Зберегти графік RMS у figures", use_container_width=True):
                saved_path = save_figure_to_png(
                    figure=fig_files_rms,
                    folder_name="figures",
                    file_name="comparison_rms.png"
                )

                st.success(f"Графік RMS збережено: {saved_path}")

            st.info(
                f"Найбільшу кількість піків виявлено у файлі `{max_peaks_file}`. "
                f"Найвище RMS після фільтрації має файл `{max_rms_file}`. "
                "Ці показники допомагають порівняти активність сигналу та рівень коливань у різних записах."
            )

        except Exception as e:
            st.error("Не вдалося виконати порівняння файлів.")
            st.write(e)

    else:
        st.warning("Файли для порівняння ще не завантажено.")

        st.info(
            "У цьому режимі потрібно завантажити кілька .txt файлів. "
            "Після цього програма автоматично побудує таблицю порівняння та графіки."
        )