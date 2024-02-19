# FFT Analysis and Audio Fingerprinting

The repository builds completely upon PyTorch and is capable of GPU optimization for computing FFT, batch processing, etc. Code is tested on Nvidia 2080Ti (CUDA 10.2). A100 and other A series GPU may face compatibility issues.

## How to run

Clone this repository and install the dependencies using `requirements.txt`

### FFT analysis

- Run `fft.ipynb` 
- Notebook has all spectrograms, ablation study and FFT codes.
- All preprocessing steps and observations related to it are in the notebook itself

### Fingerprint generation and matching

- Run all cells of `fingerprinting.ipynb`. 
- The code should automatically download the dataset and test it
- All interesting observation and code details are given in comments

## Therotical FAQs

### Round 2 

#### Describe the process of performing FFT on an audio signal. How does this transformation aid in identifying unique features within the audio for comparison purposes?

**Answer:**: The Fast Fourier Transform (FFT) is a mathematical algorithm that transforms time-domain signals into frequency-domain. For audio, this process involves sampling the signal and breaking it into bins, each representing a specific frequency component. This transformation is critical for identifying unique features such as pitch, tone, and timbre, allowing for comparison of these characteristics across different audio files.

#### Explain the importance of spectrogram resolution in audio analysis. How does the choice of parameters (e.g., window size, overlap) affect the outcome of your comparisons?

**Answer:**: Spectrogram resolution determines how precisely the frequency and time information is represented. A larger window size increases frequency resolution but decreases time resolution, and vice versa. Overlap between windows allows for smoother transitions and more data points for analysis. The choice of these parameters affects the clarity of the audio features and the accuracy of comparisons.

#### Discuss the challenges in automating the comparison of audio files, especially when dealing with alterations. How did you address these challenges in your code?

**Answer:**: Automating the comparison of audio files, particularly with alterations, presents several challenges:

1. *Variation in Alterations*: Different types of alterations can affect the audio in various ways, such as changing the frequency content, adding noise, or altering the dynamic range. This requires a comparison system that can detect and quantify a wide range of changes.

2. *Consistency Across Different Audio Qualities and Formats*: Audio files can have different sample rates, bit depths, and formats, which can affect the comparison results if not standardized.

3. *Subjectivity of Audio Perception*: Audio quality and alterations can be subjective and not always perfectly quantifiable with metrics.

In the provided code snippet, the challenges are addressed as follows:

1. *Robust Metrics*: The system employs a variety of metrics to capture different aspects of the audio signal alterations, such as MSE for overall deviation, PSNR for signal fidelity, and SSIM for structural changes. These metrics can comprehensively reflect the degree of alteration in the audio file.

2. *Preprocessing*: Before comparison, audio files are processed to standardize their formats. This is implied by the use of a single sample rate parameter in the alteration functions. This standardization ensures that comparisons are not affected by differences in sample rate or format.

3. *Controlled Alterations*: The code applies specific and controlled alterations to the audio waveform, which makes the impact of these alterations predictable and quantifiable. For instance, the 'minor' alteration applies a bass boost, 'moderate' adds noise and filters the band, and 'strong' applies a lowpass filter. Understanding the exact nature of these alterations allows the metrics to be interpreted accurately.

4. *Handling Noise*: The code adds noise in a controlled manner (in the 'moderate' alteration), allowing the comparison metrics to account for the added noise in relation to the original signal.

5. *Frequency-specific Alterations*: By applying specific filters, such as a bandpass for 'moderate' and lowpass for 'strong' alterations, the system can more accurately assess how such frequency-specific changes affect the audio quality.

To further improve the automation, the system could incorporate machine learning techniques to learn from a dataset of audio alterations and predict the alteration type or quantify the degree of alteration more accurately. Additionally, audio fingerprinting could be employed to detect if two audio files are perceptually similar despite technical alterations.

#### In creating the scoring system, what factors did you consider to quantify the degree of alteration accurately? How does your system handle variations in audio quality or format?

**Answer:**: In creating the scoring system to quantify the degree of alteration accurately, the system would consider the following factors for applying audio alterations:

1. *Magnitude of Alteration*: The scoring system would use metrics such as Mean Squared Error (MSE) to measure the magnitude of changes from the original waveform. This reflects the intensity of alterations like bass boost, noise addition, and frequency filtering.

2. *Frequency Content Changes*: Since different alterations affect the frequency content in distinct ways, the scoring system would use metrics like Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) to evaluate how the harmonic and tonal characteristics are affected.

3. *Cosine Similarity*: This metric would assess how the direction of the audio waveform vector changes with each alteration, which can indicate shifts in the overall shape of the audio spectrum.

4. *KL Divergence*: This would measure how the probability distribution of the audio signal's amplitude or power spectrum deviates from the original, which is especially relevant for the noise addition in moderate alterations.

5. *Histogram Correlation*: This would evaluate how the overall statistical distribution of the waveform's amplitude changes, remaining robust against alterations that preserve the overall shape but change specific features.

To handle variations in audio quality or format, the system would perform preprocessing steps such as:

- *Normalization*: Adjusting the waveform's amplitude to a standard level to ensure consistent measurement across different recordings.
- *Resampling*: Converting all audio files to a common sample rate to mitigate the impact of sample rate variations.
- *Format Standardization*: Converting all files to a uniform audio format before analysis, to avoid discrepancies due to compression artifacts or format-specific features.

By using these metrics and preprocessing steps, the scoring system aims to provide a reliable quantification of the degree of alteration, making it possible to compare audio files in a consistent and objective manner regardless of their initial quality or format.

#### Describe any advanced data visualization techniques you employed to present the comparison results. What insights can these visualizations provide to someone analyzing the audio files?

**Answer:**: Techniques like spectrograms, wavelet transforms, or multi-dimensional scaling provide visual representations of audio features and differences. These visualizations can help in identifying patterns, anomalies, or alterations in the audio by making them visible, which might not be evident in the raw audio or through basic statistical measures.


### FFT Analysis

#### How does the Fast Fourier Transform (FFT) differ from the Discrete Fourier Transform (DFT) in terms of computation? Why is FFT preferred for large datasets?

**Answer:** FFT is an algorithm that computes the DFT of a sequence, or its inverse, more efficiently. While DFT directly applies the definition of the Fourier transform, requiring \(O(N^2)\) operations (where \(N\) is the number of data points), FFT reduces this complexity to \(O(N \log N)\) by exploiting symmetries in the DFT's formula. This efficiency makes FFT vastly preferred for large datasets, as it significantly reduces computation time.

#### Describe a scenario in data science, other than audio processing, where FFT could be a useful tool. What kind of insights or transformations might FFT provide in your chosen scenario?

**Answer:** In computer vision, a specific application of FFT (Fast Fourier Transform) is in image compression and enhancement within satellite image analysis for environmental monitoring. Using FFT allows for the transformation of satellite images into the frequency domain, enabling:

- *Noise Reduction:* Identification and isolation of noise components to enhance image clarity.
- *Image Compression:* Reduction of redundant frequency components to compress images without significantly affecting visual quality.
- *Feature Enhancement:* Amplification of specific frequency bands to highlight environmental features like deforestation areas, water bodies, or urban expansion.

#### Can you explain the role of Fast Fourier Transform (FFT) in audio signal processing? How is FFT utilized for transforming audio data into a format suitable for analysis and pattern recognition?

**Answer:** In audio signal processing, FFT is used to convert time-domain signals into frequency-domain representations. This transformation allows for the analysis of the audio's spectral content—identifying different frequencies present in the signal. FFT enables feature extraction for sound classification, noise reduction, and the enhancement of audio signals, which are critical for applications in speech recognition, music analysis, and audio compression.

#### Describe the purpose of windowing in signal processing. What are the effects of different window types (e.g., Hamming, Hanning) on FFT output?

**Answer:** Windowing is used in signal processing to mitigate the effects of spectral leakage by tapering the signal at its boundaries before applying FFT. Different window types, like Hamming and Hanning, affect the FFT output by altering the trade-off between the main lobe width and side lobe levels. Hamming window reduces side lobe levels at the expense of slightly wider main lobes, improving frequency resolution. Hanning window provides a smoother tapering, which is beneficial for analyzing signals where minimal side lobe amplitude is crucial.

#### In a spectrogram, what does the intensity of a color at a specific point signify? How would you differentiate noise from meaningful data in a spectrogram?

**Answer:** In a spectrogram, the intensity of a color at a specific point represents the magnitude (or power) of a frequency component at a particular time. Brighter colors indicate higher energy or amplitude at that frequency and time. Differentiating noise from meaningful data involves analyzing the consistency, pattern, and spread of spectral components; noise often appears as random, sporadic, or uniformly distributed energy across frequencies, whereas meaningful signals display structured, predictable patterns. If there is a noise prior, we can remove it via designing a band pass, high/low pass filters. 

#### Describe how spectrograms are used in sound recognition systems. What features within a spectrogram are typically analyzed for recognizing different sounds or music?

**Answer:** Spectrograms are used in sound recognition systems to visualize and analyze the frequency spectrum of sounds over time. Features such as harmonic patterns, formant frequencies, and temporal variations in intensity are analyzed. The presence, absence, and relationships of these features enable the differentiation of sounds, speech, and music. Machine learning algorithms can classify these patterns to recognize specific sounds or pieces of music.

#### Discuss common challenges in audio signal processing using FFT, such as dealing with noisy data or varying signal lengths.

**Answer:** Common challenges include:
- **Noisy data:** Noise can obscure relevant frequencies, making feature extraction difficult. Techniques like noise reduction and filtering are essential.
- **Varying signal lengths:** Different lengths of audio signals can affect the FFT's ability to analyze frequencies consistently. Solutions include zero-padding shorter signals or segmenting longer signals into uniform lengths for analysis.

### Audio Fingerprinting

#### What is audio fingerprinting, and how is it applied in the context of recognizing and matching sound data? Discuss the process and its key components.

**Answer:** Audio fingerprinting is a technique used to uniquely identify a piece of audio content by extracting and encoding distinctive features from the sound. This process involves analyzing the audio signal to generate a compact, unique identifier (the fingerprint) that can be efficiently compared against a database of known fingerprints. Key components include feature extraction (identifying unique spectral or temporal characteristics), fingerprint encoding, and a matching algorithm to compare and identify audio samples.

#### In sound signal processing, what strategies can be employed to handle challenges like background noise and variations in audio quality during real-time analysis?

**Answer:** Strategies include:
- **Noise reduction techniques:** Applying filters and algorithms to reduce background noise without losing important signal details.
- **Robust feature extraction:** Identifying features that are minimally affected by noise and quality variations.
- **Machine learning models:** Training models on diverse datasets to improve recognition performance under various conditions.

#### Considering a hypothetical large-scale audio recognition system, what factors should be considered to scale the processing of audio data efficiently and effectively?
**Answer:** Factors include:

- **Computational efficiency:** Optimizing algorithms and selecting hardware capable of handling large-scale computations.
- **Database management:** Efficiently indexing and querying fingerprints to reduce lookup times.
- **Distributed processing:** Leveraging cloud computing and distributed architectures to parallelize processing and data storage.
- **Adaptability:** Incorporating feedback mechanisms to continuously improve accuracy and adapt to new sounds or noise conditions.

#### Outside of the intended theoretical approaches above, if you would like to take another approach, please provide any reasoning and additional context necessary with your answer.

**Answer:** An example could be leveraging deep learning for direct audio pattern recognition without traditional fingerprinting (like MFCC), exploiting neural networks' ability to learn complex patterns and perform end-to-end audio classification. This approach can potentially reduce the need for explicit feature extraction and offer superior performance in recognizing complex sounds and audio quality variations.

##### Submitted by: [Rohit Lal](https://rohitlal.net/)