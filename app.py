import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from src.export_utils import save_dataframe_to_csv, save_figure_to_png
from src.data_loader import read_signal_file
from src.signal_processing import (
    bandpass_filter,
    calculate_spectrum,
    calculate_rms
)
from src.analysis import analyze_file


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

selected_signal = st.sidebar.selectbox(
    "Оберіть сигнал для аналізу",
    ["signal_1", "signal_2", "signal_3"],
    index=1
)

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

            tab_overview, tab_signal, tab_compare, tab_fragment, tab_spectrum, tab_filter, tab_peaks, tab_stats = st.tabs(
                [
                    "Огляд",
                    "Графіки",
                    "Порівняння сигналів",
                    "Фрагмент",
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

            with tab_signal:
                st.subheader("Візуалізація сигналів")

                graph_col1, graph_col2 = st.columns(2)

                with graph_col1:
                    st.markdown("#### Усі сигнали")

                    fig_all, ax_all = plt.subplots(figsize=(8, 4))

                    ax_all.plot(df["time_seconds"], df["signal_1"], label="signal_1")
                    ax_all.plot(df["time_seconds"], df["signal_2"], label="signal_2")
                    ax_all.plot(df["time_seconds"], df["signal_3"], label="signal_3")

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
                        label=selected_signal
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
                    ax_s1.plot(df_fragment["time_seconds"], df_fragment["signal_1"])
                    ax_s1.set_title("signal_1")
                    ax_s1.set_xlabel("Час, с")
                    ax_s1.set_ylabel("Амплітуда")
                    ax_s1.grid(True)
                    st.pyplot(fig_s1)

                with compare_col2:
                    st.markdown("#### signal_2")

                    fig_s2, ax_s2 = plt.subplots(figsize=(5, 3))
                    ax_s2.plot(df_fragment["time_seconds"], df_fragment["signal_2"])
                    ax_s2.set_title("signal_2")
                    ax_s2.set_xlabel("Час, с")
                    ax_s2.set_ylabel("Амплітуда")
                    ax_s2.grid(True)
                    st.pyplot(fig_s2)

                with compare_col3:
                    st.markdown("#### signal_3")

                    fig_s3, ax_s3 = plt.subplots(figsize=(5, 3))
                    ax_s3.plot(df_fragment["time_seconds"], df_fragment["signal_3"])
                    ax_s3.set_title("signal_3")
                    ax_s3.set_xlabel("Час, с")
                    ax_s3.set_ylabel("Амплітуда")
                    ax_s3.grid(True)
                    st.pyplot(fig_s3)

                st.warning(
                    "На цьому етапі програма не визначає автоматично, де саме PCG. "
                    "Попередньо signal_2 використовується як основний PCG-сигнал."
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
                    label=f"{selected_signal} ({start_time:.1f}–{end_time:.1f} с)"
                )

                ax_fragment.set_title(f"Фрагмент сигналу: {selected_signal}")
                ax_fragment.set_xlabel("Час, с")
                ax_fragment.set_ylabel("Амплітуда")
                ax_fragment.legend()
                ax_fragment.grid(True)

                st.pyplot(fig_fragment)

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
                    spectrum[spectrum_mask]
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
                    f"спостерігається приблизно на частоті {dominant_frequency_visible:.2f} Гц."
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
                        label=signal_name
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
                    alpha=0.7
                )

                ax_filter_full.plot(
                    df["time_seconds"],
                    df["filtered_signal"],
                    label="Відфільтрований сигнал",
                    alpha=0.9
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
                    alpha=0.7
                )

                ax_filter_fragment.plot(
                    df_filtered_fragment["time_seconds"],
                    df_filtered_fragment["filtered_signal"],
                    label="Відфільтрований фрагмент",
                    alpha=0.9
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
                    label="Відфільтрований сигнал"
                )

                ax_peaks_full.scatter(
                    peak_times,
                    peak_values,
                    label="Знайдені піки",
                    marker="o"
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
                    label="Відфільтрований фрагмент"
                )

                ax_peaks_fragment.scatter(
                    fragment_peak_times,
                    fragment_peak_values,
                    label="Піки у фрагменті",
                    marker="o"
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
                comparison_df["Кількість піків"]
            )

            ax_files_peaks.set_title("Порівняння кількості піків у файлах")
            ax_files_peaks.set_xlabel("Файл")
            ax_files_peaks.set_ylabel("Кількість піків")
            ax_files_peaks.tick_params(axis="x", rotation=25)
            ax_files_peaks.grid(True, axis="y")

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
                comparison_df["RMS після фільтрації"]
            )

            ax_files_rms.set_title("Порівняння RMS після фільтрації")
            ax_files_rms.set_xlabel("Файл")
            ax_files_rms.set_ylabel("RMS")
            ax_files_rms.tick_params(axis="x", rotation=25)
            ax_files_rms.grid(True, axis="y")

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