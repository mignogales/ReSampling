from extras.data_loader import convert_tsf_to_dataframe

if __name__ == "__main__":
    # data_path = "/home/mnogales/Projects/ReSampling/datasets/Wind/wind_farms_minutely_dataset_with_missing_values.tsf"
    # loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(data_path)
    # print(loaded_data.head())

    # # print the series lenght. df has columns: 'series_name', 'start_time', 'series_value' last is a list
    # print(f"Number of series: {len(loaded_data)}")
    # print(f"Frequency: {frequency}")
    # print(f"Forecast horizon: {forecast_horizon}")
    # print(f"Contain missing values: {contain_missing_values}")
    # print(f"Contain equal length: {contain_equal_length}")

    # # print the length of all series
    # for index, row in loaded_data.iterrows():
    #     series_name = row['series_name']
    #     series_length = len(row['series_value'])
    #     print(f"Series: {series_name}, Length: {series_length}")


    # now open datasets/Electricity/australian_electricity_demand_dataset.tsf
    # data_path = "/home/mnogales/Projects/ReSampling/datasets/Electricity/australian_electricity_demand_dataset.tsf"
    data_path = "/home/mnogales/Projects/ReSampling/datasets/Solar/solar_10_minutes_dataset.tsf"
    loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(data_path)
    print(loaded_data.head())   

    period = 10 # minutes

    print(f"Number of series: {len(loaded_data)}")
    print(f"Frequency: {frequency}")
    print(f"Forecast horizon: {forecast_horizon}")
    print(f"Contain missing values: {contain_missing_values}")
    print(f"Contain equal length: {contain_equal_length}") 

    # print the length of all series
    for index, row in loaded_data.iterrows():
        series_name = row['series_name']
        series_length = len(row['series_value'])
        print(f"Series: {series_name}, Length: {series_length}")

    # show me the spectra of the electricity dataset
    import matplotlib.pyplot as plt
    import numpy as np

    series_to_plot = loaded_data.iloc[0]['series_value']
    # susbtract the mean
    series_to_plot = series_to_plot - np.mean(series_to_plot)
    series_fft = np.fft.fft(series_to_plot)
    freq = np.fft.fftfreq(len(series_to_plot))  

    # make the plot in log scale
    plt.figure(figsize=(12, 6))
    plt.plot(freq, np.abs(series_fft))
    plt.title('FFT of Electricity Series')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.xlim(0, 0.5)  # Limit x-axis for better
    plt.yscale('log')
    plt.grid()
    plt.savefig("electricity_spectrum.png")

    # compute the spectogram
    from scipy.signal import spectrogram
    f, t, Sxx = spectrogram(series_to_plot, fs=1/period)
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.colorbar(label='Intensity [dB]')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [minutes]')
    plt.title('Spectrogram of Electricity Series')
    plt.savefig("electricity_spectrogram.png")

    # plot the first 500 points of the series
    plt.figure(figsize=(12, 6))
    plt.plot(loaded_data.iloc[0]['series_value'][:500])
    plt.title('First 500 points of Electricity Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid()
    plt.savefig("electricity_first_500_points.png")