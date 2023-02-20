"""
Artifact Subspace Reconstruction.
This code is an adaptation based on the code from the repository
https://github.com/nbara/python-meegkit
Specific parts of the repo https://github.com/DiGyt/asrpy were also
used.
Author of MEEGkit: Nicolas Barascud
Author of ASRpy: Dirk Gütlin
Author of this repository: Mauricio Cespedes Tenorio
"""
import logging

import numpy as np
from scipy import linalg, signal
from statsmodels.robust.scale import mad
import psutil

from .utils import block_covariance, nonlinear_eigenspace
from .utils.asr import (geometric_median, fit_eeg_distribution, yulewalk,
                        yulewalk_filter, ma_filter)

try:
    import pyriemann
except ImportError:
    pyriemann = None


class ASR():
    """Artifact Subspace Reconstruction.

    Artifact subspace reconstruction (ASR) is an automatic, online,
    component-based artifact removal method for removing transient or
    large-amplitude artifacts in multi-channel EEG recordings [1]_.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the data, in Hz.

    The following are optional parameters (the key parameter of the method is
    the ``cutoff``):
    
    burst_rejection: bool
        If True reject portions of data containing burst instead of 
        correcting them using ASR. Default is True.
    cutoff: float
        Standard deviation cutoff for rejection. X portions whose variance
        is larger than this threshold relative to the calibration data are
        considered missing data and will be removed. The most aggressive value
        that can be used without losing too much EEG is 2.5. A quite
        conservative value would be 5 (default=5).
    blocksize : int
        Block size for calculating the robust data covariance and thresholds,
        in samples; allows to reduce the memory and time requirements of the
        robust estimators by this factor (down to Channels x Channels x Samples
        x 16 / Blocksize bytes) (default=10).
    win_len : float
        Window length (s) that is used to check the data for artifact content.
        This is ideally as long as the expected time scale of the artifacts but
        not shorter than half a cycle of the high-pass filter that was used
        (default=1).
    win_overlap : float
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are going
        to be missed, but is slower (default=0.66).
    max_dropout_fraction : float
        Maximum fraction of windows that can be subject to signal dropouts
        (e.g., sensor unplugged), used for threshold estimation (default=0.1).
    min_clean_fraction : float
        Minimum fraction of windows that need to be clean, used for threshold
        estimation (default=0.25).
    method : {'riemann', 'euclid'}
        Method to use. If riemann, use the riemannian-modified version of
        ASR [2]_.
    memory : float
        Memory size (s), regulates the number of covariance matrices to store.
    estimator : str in {'scm', 'lwf', 'oas', 'mcd'}
        Covariance estimator (default: 'scm' which computes the sample
        covariance). Use 'lwf' if you need regularization (requires pyriemann).
    maxmem : Amount of memory to use. See asr_process for more information.

    Attributes
    ----------
    ``state_`` : dict
        Initial state of the ASR filter.
    ``zi_``: array, shape=(n_channels, filter_order)
        Filter initial conditions.
    ``cov_`` : array, shape=(channels, channels)
        Previous covariance matrix.
    ``state_`` : dict
        Previous ASR parameters (as derived by :func:`asr_calibrate`) for
        successive calls to :meth:`transform`. Required fields are:

        - ``M`` : Mixing matrix
        - ``T`` : Threshold matrix
        - ``R`` : Reconstruction matrix (array | None)

    References
    ----------
    .. [1] Kothe, C. A. E., & Jung, T. P. (2016). U.S. Patent Application No.
       14/895,440. https://patents.google.com/patent/US20160113587A1/en
    .. [2] Blum, S., Jacobsen, N. S. J., Bleichner, M. G., & Debener, S.
       (2019). A Riemannian Modification of Artifact Subspace Reconstruction
       for EEG Artifact Handling. Frontiers in Human Neuroscience, 13.
       https://doi.org/10.3389/fnhum.2019.00141

    """

    def __init__(self, sfreq, burst_rejection = True, cutoff=5, blocksize=100, win_len=0.5,
                 win_overlap=0.66, max_dropout_fraction=0.1,
                 min_clean_fraction=0.25, method='euclid',
                 estimator='scm', maxmem = None):

        if pyriemann is None and method == 'riemann':
            logging.warning('Need pyriemann to use riemannian ASR flavor.')
            method = 'euclid'

        self.cutoff = cutoff
        self.burst_rejection = burst_rejection
        self.blocksize = blocksize
        self.win_len = win_len
        self.win_overlap = win_overlap
        self.max_dropout_fraction = max_dropout_fraction
        self.min_clean_fraction = min_clean_fraction
        self.max_bad_chans = 0.3
        self.method = method
        self.memory = int(2 * sfreq)  # smoothing window for covariances
        self.sample_weight = np.geomspace(0.05, 1, num=self.memory + 1)
        self.sfreq = sfreq
        self.estimator = estimator

        self.reset()

    def reset(self):
        self.cov_ = None
        self.zi_ = None
        self.state_ = {}
        self._counter = []
        self._fitted = False

    def fit(self, raw_iEEG, channels = [], start=0, stop=None, 
            return_signal = False):
        """Calibration for the Artifact Subspace Reconstruction method.

        The input to this data is a multi-channel time series of calibration
        data. In typical uses the calibration data is clean resting EEG data of
        data if the fraction of artifact content is below the breakdown point
        of the robust statistics used for estimation (50% theoretical, ~30%
        practical). If the data has a proportion of more than 30-50% artifacts
        then bad time windows should be removed beforehand. This data is used
        to estimate the thresholds that are used by the ASR processing function
        to identify and remove artifact components.

        The calibration data must have been recorded for the same cap design
        from which data for cleanup will be recorded, and ideally should be
        from the same session and same subject, but it is possible to reuse the
        calibration data from a previous session and montage to the extent that
        the cap is placed in the same location (where loss in accuracy is more
        or less proportional to the mismatch in cap placement).

        Parameters
        ----------
        raw_iEEG : array, shape=(n_channels, n_samples)
            The calibration data should have been high-pass filtered (for
            example at 0.5Hz or 1Hz using a Butterworth IIR filter), and be
            reasonably clean not less than 30 seconds (this method is typically
            used with 1 minute or more).
        channels : list
            Indexes of channels used to fit the ASR. All channels should be of
            the same lenght. Defaults to use all the channels.
        start : int
            The first sample to use for fitting the data. Defaults to 0.
        stop : int | None
            The last sample to use for fitting the data. If `None`, all 
            samples after `start` will be used for fitting. Defaults to None.
        return_signal : Bool
            If True, the method will return the variables `clean` (the cropped
             dataset which was used to fit the ASR) and `sample_mask` (a
             logical mask of which samples were included/excluded from fitting
             the ASR). Defaults to False.

        """
        if raw_iEEG.ndim > 2:
            raise Exception('The shape of the iEEG data must be (n_channels, n_samples)')
        if len(channels)>0:
            X = raw_iEEG[channels,:]
        else:
            X = raw_iEEG

        # Find artifact-free windows first
        clean, sample_mask = clean_windows(
            X,
            sfreq=self.sfreq,
            win_len=self.win_len,
            win_overlap=self.win_overlap,
            max_bad_chans=self.max_bad_chans,
            min_clean_fraction=self.min_clean_fraction,
            max_dropout_fraction=self.max_dropout_fraction)

        # Perform calibration
        M, T = asr_calibrate(
            clean,
            sfreq=self.sfreq,
            cutoff=self.cutoff,
            blocksize=self.blocksize,
            win_len=self.win_len,
            win_overlap=self.win_overlap,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            method=self.method,
            estimator=self.estimator)

        self.state_ = dict(M=M, T=T, R=None)
        self._fitted = True

        return clean, sample_mask

    def transform(self, raw_iEEG, channels = [], lookahead=None, stepsize=32, 
                  maxdims=0.66, return_states=False, mem_splits=3):
        """Apply Artifact Subspace Reconstruction.

        Parameters
        ----------
        raw_iEEG : array, shape=([n_trials, ]n_channels, n_samples)
            Raw data.
        channels : list
            Indexes of channels used to fit the ASR. All channels should be of
            the same lenght. Defaults to use all the channels.
        lookahead : float
            Amount of look-ahead that the algorithm should use (in seconds). 
            This value should be between 0 (no lookahead) and WindowLength/2 
            (optimal lookahead). The recommended value is WindowLength/2. 
            Default: WindowLength/2
            
            Note: Other than in `asr_process`, the signal will be readjusted 
            to eliminate any temporal jitter and automatically readjust it to 
            the correct time points. Zero-padding will be applied to the last 
            `lookahead` portion of the data, possibly resulting in inaccuracies 
            for the final `lookahead` seconds of the recording.
        stepsize : int
            The steps in which the algorithm will be updated. The larger this
            is, the faster the algorithm will be. The value must not be larger
            than WindowLength * SamplingRate. The minimum value is 1 (update
            for every sample) while a good value would be sfreq//3. Note that
            an update is always performed also on the first and last sample of
            the data chunk. Default: 32
        max_dims : float, int
            Maximum dimensionality of artifacts to remove. This parameter
            denotes the maximum number of dimensions which can be removed from
            each segment. If larger than 1, `int(max_dims)` will denote the
            maximum number of dimensions removed from the data. If smaller
            than 1, `max_dims` describes a fraction of total dimensions.
            Defaults to 0.66.
        return_states : bool
            If True, returns a dict including the updated states {"M":M,
            "T":T, "R":R, "Zi":Zi, "cov":cov, "carry":carry}. Defaults to
            False.
        
        Returns
        -------
        out : array, shape=(n_channels, n_samples)
            Filtered data.
        """
        # For multiple trials data
        if raw_iEEG.ndim == 3:
            if raw_iEEG.shape[0] == 1:  # single epoch case
                out = self.transform(raw_iEEG[0])
                return out[None, ...]
            else:
                outs = [self.transform(x) for x in raw_iEEG]
                return np.stack(outs, axis=0)

        if not self._fitted:
                logging.warning('ASR is not fitted ! Returning unfiltered data.')
                return raw_iEEG

        # Extract the data
        if len(channels)>0:
            X = raw_iEEG[channels,:]
        else:
            X = raw_iEEG
        
        # add lookahead padding at the end
        if lookahead == None:
            lookahead = self.win_len/2
        lookahead_samples = int(self.sfreq * lookahead)
        Y = np.concatenate([X,
                            np.zeros([X.shape[0], lookahead_samples])],
                            axis=1)
        
        # Exponential covariance weights – the most recent covariance has a
        # weight of 1, while the oldest one in memory has a weight of 5%
        weights = [1, ]
        for c in np.cumsum(self._counter[1:]):
            weights = [self.sample_weight[-c]] + weights
            
        # apply ASR
        if return_states:
            Y, self.state_ = asr_process(Y, self.sfreq, self.state_, self.win_len,
                                        lookahead, stepsize, maxdims,
                                        self.zi_, self.cov_,
                                        return_states, self.method, 
                                        weights, mem_splits)
        else:
            Y = asr_process(Y, self.sfreq, self.state_, self.win_len,
                            lookahead, stepsize, maxdims,
                            self.zi_, self.cov_,
                            return_states, self.method, weights, mem_splits)
        
        # remove lookahead portion from start
        print(Y.size)
        Y = Y[:, lookahead_samples:] # Review this!!
        
        sample_mask = np.sum(np.abs(X-Y), axis=0) < 1e-8

        # if self.burst_rejection:
            

        # else:
        return Y, sample_mask


def clean_windows(X, sfreq, max_bad_chans=0.2, zthresholds=[-3.5, 5],
                  win_len=.5, win_overlap=0.66, min_clean_fraction=0.25,
                  max_dropout_fraction=0.1, show=False):
    """Remove periods with abnormally high-power content from continuous data.

    This function cuts segments from the data which contain high-power
    artifacts. Specifically, only windows are retained which have less than a
    certain fraction of "bad" channels, where a channel is bad in a window if
    its power is above or below a given upper/lower threshold (in standard
    deviations from a robust estimate of the EEG power distribution in the
    channel).

    Parameters
    ----------
    X : array, shape=(n_channels, n_samples)
        Continuous data set, assumed to be appropriately high-passed (e.g. >
        1Hz or 0.5Hz - 2.0Hz transition band)
    max_bad_chans : float
        The maximum number or fraction of bad channels that a retained window
        may still contain (more than this and it is removed). Reasonable range
        is 0.05 (very clean output) to 0.3 (very lax cleaning of only coarse
        artifacts) (default=0.2).
    zthresholds : 2-tuple
        The minimum and maximum standard deviations within which the power of
        a channel must lie (relative to a robust estimate of the clean EEG
        power distribution in the channel) for it to be considered "not bad".
        (default=[-3.5, 5]).
    save_fig : bool
        Defines whether it is desired to save an output figure of the cleaned
        signal. Default: False.

    The following are detail parameters that usually do not have to be tuned.
    If you can't get the function to do what you want, you might consider
    adapting these to your data.

    win_len : float
        Window length that is used to check the data for artifact content.
        This is ideally as long as the expected time scale of the artifacts
        but not shorter than half a cycle of the high-pass filter that was
        used. Default: 1.
    win_overlap : float
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are
        going to be missed, but is slower (default=0.66).
    min_clean_fraction : float
        Minimum fraction that needs to be clean. This is the minimum fraction
        of time windows that need to contain essentially uncontaminated EEG.
        (default=0.25)
    max_dropout_fraction : float
        Maximum fraction that can have dropouts. This is the maximum fraction
        of time windows that may have arbitrarily low amplitude (e.g., due to
        the sensors being unplugged) (default=0.1).

    Returns
    -------
    clean : array, shape=(n_channels, n_samples)
        Dataset with bad time periods removed.
    sample_mask : boolean array, shape=(1, n_samples)
        Mask of retained samples (logical array).

    """
    assert 0 < max_bad_chans < 1, "max_bad_chans must be a fraction !"

    # set internal variables
    truncate_quant = [0.0220, 0.6000]
    step_sizes = [0.01, 0.01]
    shape_range = np.arange(1.7, 3.5, 0.15)
    max_bad_chans = np.round(X.shape[0] * max_bad_chans)

    # set data indices
    [nc, ns] = X.shape
    N = int(win_len * sfreq)
    offsets = np.round(np.arange(0, ns - N, (N * (1 - win_overlap))))
    offsets = offsets.astype(int)
    logging.debug('[ASR] Determining channel-wise rejection thresholds')

    wz = np.zeros((nc, len(offsets)))
    for ichan in range(nc):

        # compute root mean squared amplitude
        x = X[ichan, :] ** 2
        Y = np.array([np.sqrt(np.sum(x[o:o + N]) / N) for o in offsets])

        # fit a distribution to the clean EEG part
        mu, sig, alpha, beta = fit_eeg_distribution(
            Y, min_clean_fraction, max_dropout_fraction, truncate_quant,
            step_sizes, shape_range)
        # calculate z scores
        wz[ichan] = (Y - mu) / sig

    # sort z scores into quantiles
    wz[np.isnan(wz)] = np.inf  # Nan to inf
    swz = np.sort(wz, axis=0)

    # determine which windows to remove based on the superior and inferior
    # z-thresholds
    if np.max(zthresholds) > 0:
        mask1 = swz[-(int(max_bad_chans) + 1), :] > np.max(zthresholds)
    if np.min(zthresholds) < 0:
        mask2 = (swz[1 + int(max_bad_chans - 1), :] < np.min(zthresholds))

    # Remove criteria based on Median Absolute Deviation 
    bad_by_mad = mad(wz, c=1, axis=0) < .1
    # Remove criteria based on Standard Deviation
    bad_by_std = np.std(wz, axis=0) < .1
    mask3 = np.logical_or(bad_by_mad, bad_by_std)

    # combine the three masks
    remove_mask = np.logical_or.reduce((mask1, mask2, mask3))
    removed_wins = np.where(remove_mask)[0]

    # reconstruct the samples to remove
    sample_maskidx = []
    for i, win in enumerate(removed_wins):
        if i == 0:
            sample_maskidx = np.arange(offsets[win], offsets[win] + N)
        else:
            sample_maskidx = np.r_[(sample_maskidx,
                                    np.arange(offsets[win], offsets[win] + N))]

    # delete the bad chunks from the data
    sample_mask2remove = np.unique(sample_maskidx)
    if sample_mask2remove.size:
        clean = np.delete(X, sample_mask2remove, axis=1)
        sample_mask = np.ones((1, ns), dtype=bool)
        sample_mask[0, sample_mask2remove] = False
    else:
        clean = X
        sample_mask = np.ones((1, ns), dtype=bool)

    if show:
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(nc, sharex=True, figsize=(8, 5))
        times = np.arange(ns) / float(sfreq)
        for i in range(nc):
            ax[i].fill_between(times, 0, 1, where=sample_mask.flat,
                               transform=ax[i].get_xaxis_transform(),
                               facecolor='none', hatch='...', edgecolor='k',
                               label='selected window')
            ax[i].plot(times, X[i], lw=.5, label='EEG')
            ax[i].set_ylim([-50, 50])
            # ax[i].set_ylabel(raw.ch_names[i])
            ax[i].set_yticks([])
        ax[i].set_xlabel('Time (s)')
        ax[i].set_ylabel(f'ch{i}')
        ax[0].legend(fontsize='small', bbox_to_anchor=(1.04, 1),
                     borderaxespad=0)
        plt.subplots_adjust(hspace=0, right=0.75)
        plt.suptitle('Clean windows')
        # plt.show() Change to save in HTML

    return clean, sample_mask


def asr_calibrate(X, sfreq, cutoff=5, blocksize=10, win_len=0.5,
                  win_overlap=0.66, max_dropout_fraction=0.1,
                  min_clean_fraction=0.25, method='euclid', estimator='scm',
                  maxmem=None):
    """Calibration function for the Artifact Subspace Reconstruction method.

    The input to this data is a multi-channel time series of calibration data.
    In typical uses the calibration data is clean resting EEG data of ca. 1
    minute duration (can also be longer). One can also use on-task data if the
    fraction of artifact content is below the breakdown point of the robust
    statistics used for estimation (50% theoretical, ~30% practical). If the
    data has a proportion of more than 30-50% artifacts then bad time windows
    should be removed beforehand. This data is used to estimate the thresholds
    that are used by the ASR processing function to identify and remove
    artifact components.

    The calibration data must have been recorded for the same cap design from
    which data for cleanup will be recorded, and ideally should be from the
    same session and same subject, but it is possible to reuse the calibration
    data from a previous session and montage to the extent that the cap is
    placed in the same location (where loss in accuracy is more or less
    proportional to the mismatch in cap placement).

    The calibration data should have been high-pass filtered (for example at
    0.5Hz or 1Hz using a Butterworth IIR filter).

    Parameters
    ----------
    X : array, shape=([n_trials, ]n_channels, n_samples)
        *zero-mean* (e.g., high-pass filtered) and reasonably clean EEG of not
        much less than 30 seconds (this method is typically used with 1 minute
        or more).
    sfreq : float
        Sampling rate of the data, in Hz.
    cutoff: float
        Standard deviation cutoff for rejection. X portions whose variance is
        larger than this threshold relative to the calibration data are
        considered missing data and will be removed. The most aggressive value
        that can be used without losing too much EEG is 5 (according to EEGlab
        original implementation).
    blocksize : int
        Block size for calculating the robust data covariance and thresholds,
        in samples; allows to reduce the memory and time requirements of the
        robust estimators by this factor (down to n_chans x n_chans x n_samples
        x 16 / blocksize bytes) (default=10).
    win_len : float
        Window length that is used to check the data for artifact content. This
        is ideally as long as the expected time scale of the artifacts but
        short enough to allow for several 1000 windows to compute statistics
        over (default=0.5).
    win_overlap : float
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are going
        to be missed, but is slower (default=0.66).
    max_dropout_fraction : float
        Maximum fraction of windows that can be subject to signal dropouts
        (e.g., sensor unplugged), used for threshold estimation (default=0.1).
    min_clean_fraction : float
        Minimum fraction of windows that need to be clean, used for threshold
        estimation (default=0.25).
    method : {'euclid', 'riemann'}
        Metric to compute the covariance matrix average.
    maxmem : The maximum amount of memory used by the algorithm when processing a long chunk with
                many channels, in MB. The recommended value is 64 Mb.
                default: Available memory calculated through psutil package. 

    Returns
    -------
    M : array
        Mixing matrix.
    T : array
        Threshold matrix.

    """
    ##### Blocksize is calculated differently in MATLAB code!! Please check and fix!
    logging.debug('[ASR] Calibrating...')

    # set number of channels and number of samples
    [nc, ns] = X.shape

    # Check if chosen blocksize will not saturate memory
    if maxmem==None:
        # Set to available memory
        maxmem = psutil.virtual_memory()[1]
    # Requested memory. Asuming possible complex values and taking in
    # consideration the padding
    req_mem = nc*nc*16*(ns+blocksize)/blocksize
    if req_mem>maxmem:
        try:
            # Recalculate the blocksize taking 80% of available mem
            blocksize = int(-(16*nc*nc*ns)/(16*nc*nc - maxmem*0.8))
        except:
            # Raise error if there's not enough memory
            raise Exception('Not enough memory to compute the operation!')
    
    # filter the data
    X, _zf = yulewalk_filter(X, sfreq, ab=None)

    # window length for calculating thresholds
    N = int(np.round(win_len * sfreq))

    # The covariance is calculated differently in MATLAB code as there's more
    # overlap in their approach
    # X is divided into epochs of size=blocksize with some overlap. The cov
    # is then calculated for each window. size(U)=(# of windows, nc, nc)
    U = block_covariance(X, window=blocksize, overlap=win_overlap,
                         estimator=estimator)
    # Calculate a median cov matrix
    if method == 'euclid':
        Uavg = geometric_median(U.reshape((-1, nc * nc)))
        Uavg = Uavg.reshape((nc, nc))
    else:  # method == 'riemann'
        Uavg = pyriemann.utils.mean.mean_covariance(U, metric='riemann')

    # get the mixing matrix M such that M@M.T = U
    M = linalg.sqrtm(np.real(Uavg))

    # Get eigenvector/eigenvalues and sort them by the eigvalues
    if method == 'riemann':
        D, Vtmp = nonlinear_eigenspace(M, nc)  # TODO
    else:
        D, Vtmp = linalg.eigh(M)
    # D, Vtmp = linalg.eigh(M)
    # D, Vtmp = nonlinear_eigenspace(M, nc)  TODO
    V = Vtmp[:, np.argsort(D)]

    ## Get the threshold matrix T
    # First convert x to PCA space, which would be kind of a 'average'
    # space calculated with the different windows
    x = np.abs(np.dot(V.T, X))
    offsets = np.arange(0, ns - N, np.round(N * (1 - win_overlap))).astype(int)

    # go through all the "channels" and fit the EEG distribution to get
    # the median 'mu' and standard deviation 'sig' for each. These are not
    # any more the channels but the data in the new PCA coordinate space. Then, 
    # each 'channel' is associated with a value in V
    mu = np.zeros(nc)
    sig = np.zeros(nc)
    for ichan in reversed(range(nc)):
        rms = x[ichan, :] ** 2
        Y = []
        for o in offsets:
            Y.append(np.sqrt(np.sum(rms[o:o + N]) / N))

        mu[ichan], sig[ichan], alpha, beta = fit_eeg_distribution(
            Y, min_clean_fraction, max_dropout_fraction)

    # Here, we are just 'weighting' V according to the metrics calculated
    # before. 
    T = np.dot(np.diag(mu + cutoff * sig), V.T)
    logging.debug('[ASR] Calibration done.')
    return M, T


def asr_process(data, sfreq, state, windowlen=0.5, lookahead=0.25, stepsize=32,
                maxdims=0.66, Zi=None, cov=None, return_states=False, method="euclid", 
                sample_weight=None, mem_splits=3):
    """Apply the Artifact Subspace Reconstruction method to a data array.
    This function is used to clean multi-channel signal using the ASR method.
    The required inputs are the data matrix and the sampling rate of the data.
    , ab=ab, axis=-1
    `asr_process` can be used if you inted to apply ASR to a simple numpy 
    array instead of a mne.io.Raw object. It is equivalent to the MATLAB 
    implementation of `asr_process` (except for some small differences 
    introduced by solvers for the eigenspace functions etc).
    Parameters
    ----------
    data : array, shape=(n_channels, n_samples)
        Raw data.
    sfreq : float
        The sampling rate of the data.
    state : dict
        Initial ASR parameters (as derived by :func:`asr_calibrate`):

        - ``M`` : Mixing matrix
        - ``T`` : Threshold matrix
        - ``R`` : Previous reconstruction matrix (array | None)
    windowlen : float
        Window length that is used to check the data for artifact content.
        This is ideally as long as the expected time scale of the artifacts
        but short enough to allow for several 1000 windows to compute
        statistics over (default=0.5).
    lookahead:
        Amount of look-ahead that the algorithm should use. Since the
        processing is causal, the output signal will be delayed by this
        amount. This value is in seconds and should be between 0 (no
        lookahead) and WindowLength/2 (optimal lookahead). The recommended
        value is WindowLength/2. Default: 0.25
    stepsize:
        The steps in which the algorithm will be updated. The larger this is,
        the faster the algorithm will be. The value must not be larger than
        WindowLength * SamplingRate. The minimum value is 1 (update for every
        sample) while a good value would be sfreq//3. Note that an update
        is always performed also on the first and last sample of the data
        chunk. Default: 32
    max_dims : float, int
        Maximum dimensionality of artifacts to remove. This parameter
        denotes the maximum number of dimensions which can be removed from
        each segment. If larger than 1, `int(max_dims)` will denote the
        maximum number of dimensions removed from the data. If smaller than 1,
        `max_dims` describes a fraction of total dimensions. Defaults to 0.66.
    Zi : array
        Previous filter conditions. Defaults to None.
    cov : array, shape=([n_trials, ]n_channels, n_channels) | None
        Covariance. If None (default), then it is computed from ``X_filt``.
        If a 3D array is provided, the average covariance is computed from
        all the elements in it. Defaults to None.
    return_states : bool
        If True, returns a dict including the updated states {"M":M, "T":T,
        "R":R, "Zi":Zi, "cov":cov, "carry":carry}. Defaults to False.
    method : {'euclid', 'riemann'}
        Metric to compute the covariance matrix average. Currently, only
        euclidean ASR is supported.
    mem_splits : int
        Split the array in `mem_splits` segments to save memory.
    Returns
    -------
    clean : array, shape=(n_channels, n_samples)
        Clean data.
    state : dict
        Output ASR parameters {"M":M, "T":T, "R":R, "Zi":Zi, "cov":cov,
        "carry":carry}.
    """
    # calculate the the actual max dims based on the fraction parameter
    if maxdims < 1:
        maxdims = np.round(len(data) * maxdims)

    # set initial filter conditions of none was passed
    if Zi is None:
        _, Zi = yulewalk_filter(data, sfreq=sfreq,
                                zi=np.ones([len(data), 8]))

    M, T, R = state.values()
    # set the number of channels
    C, S = data.shape

    # set the number of windows
    N = np.round(windowlen * sfreq).astype(int)
    P = np.round(lookahead * sfreq).astype(int)

    # interpolate a portion of the data 
    carry = np.tile(2 * data[:, 0],
                    (P, 1)).T - data[:, np.mod(np.arange(P, 0, -1), S)]
    data = np.concatenate([carry, data], axis=-1)

    # splits = np.ceil(C*C*S*8*8 + C*C*8*s/stepsize + C*S*8*2 + S*8*5)...
    splits = mem_splits  # TODO: use this for parallelization MAKE IT A PARAM FIRST

    # loop over smaller segments of the data (for memory purposes)
    last_trivial = False
    last_R = None
    for i in range(splits):

        # set the current range
        i_range = np.arange(i * S // splits,
                            np.min([(i + 1) * S // splits, S]),
                            dtype=int)

        # filter the current window with yule-walker
        X, Zi = yulewalk_filter(data[:, i_range + P], sfreq=sfreq,
                                zi=Zi)
        print(X.shape)

        # compute a moving average covariance
        if method == "riemann":
            #warnings.warn("Riemannian ASR is not yet supported. Switching back to"
                     # " Euclidean ASR.")
            cov = pyriemann.utils.mean.mean_covariance(
                cov, metric='riemann', sample_weight=sample_weight)
        else:
            Xcov, cov = \
                ma_filter(N,
                        np.reshape(np.multiply(np.reshape(X, (1, C, -1)),
                                                np.reshape(X, (C, 1, -1))),
                                    (C * C, -1)), cov)
            # Xcov, cov = \
            #     ma_filter(N,
            #               np.reshape(np.multiply(np.reshape(X, (1, C, -1)),
            #                                      np.reshape(X, (C, 1, -1))),
            #                          (C * C, -1)), cov) 
        

        # set indices at which we update the signal
        update_at = np.arange(stepsize,
                              Xcov.shape[-1] + stepsize - 2,
                              stepsize)
        update_at = np.minimum(update_at, Xcov.shape[-1]) - 1

        # set the previous reconstruction matrix if none was assigned
        if last_R is None:
            update_at = np.concatenate([[0], update_at])
            last_R = np.eye(C)

        Xcov = np.reshape(Xcov[:, update_at], (C, C, -1))

        # loop through the updating intervals
        last_n = 0
        for j in range(len(update_at) - 1):

            # get the eigenvectors/values.For method 'riemann', this should
            # be replaced with PGA/ nonlinear eigenvalues
            D, V = np.linalg.eigh(Xcov[:, :, j])

            # determine which components to keep
            keep = np.logical_or(D < np.sum((T @ V)**2, axis=0),
                                 np.arange(C) + 1 < (C - maxdims))
            
            # update the reconstruction matrix R (reconstruct artifact components using
            # the mixing matrix)
            trivial = keep.all() 
            if trivial:
                R = np.eye(C)  # trivial case
            else:
                inv = linalg.pinv(np.multiply(keep[:, np.newaxis], V.T @ M))
                R = np.real(M @ inv @ V.T)

            # apply the reconstruction
            n = update_at[j] + 1
            if (not trivial) or (not last_trivial):

                subrange = i_range[np.arange(last_n, n)]

                # generate a cosine signal
                blend_x = np.pi * np.arange(1, n - last_n + 1) / (n - last_n)
                blend = (1 - np.cos(blend_x)) / 2

                # use cosine blending to replace data with reconstructed data
                tmp_data = data[:, subrange]
                data[:, subrange] = np.multiply(blend, R @ tmp_data) + \
                                    np.multiply(1 - blend, last_R @ tmp_data) # noqa

            # set the parameters for the next iteration
            last_n, last_R, last_trivial = n, R, trivial

    # assign a new lookahead portion
    carry = np.concatenate([carry, data[:, -P:]])
    carry = carry[:, -P:]

    if return_states:
        return data[:, :-P], {"M": M, "T": T, "R": R, "Zi": Zi,
                              "cov": cov, "carry": carry}
    else:
        return data[:, :-P]