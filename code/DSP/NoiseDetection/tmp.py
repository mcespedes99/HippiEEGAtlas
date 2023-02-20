def transform(self, raw_iEEG, channels = [], lookahead=0.25, stepsize=32, 
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
        Default: 0.25
        
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
    lookahead_samples = int(self.sfreq * lookahead)
    X = np.concatenate([X,
                        np.zeros([X.shape[0], lookahead_samples])],
                        axis=1)
    
    # Exponential covariance weights – the most recent covariance has a
    # weight of 1, while the oldest one in memory has a weight of 5%
    weights = [1, ]
    for c in np.cumsum(self._counter[1:]):
        weights = [self.sample_weight[-c]] + weights
        
    # apply ASR
    if return_states:
        X, self.state_ = asr_process(X, self.sfreq, self.state_, self.win_len,
                                    lookahead, stepsize, maxdims,
                                    self.zi_, self.cov_,
                                    return_states, self.method, 
                                    weights, mem_splits)
    else:
        X = asr_process(X, self.sfreq, self.state_, self.win_len,
                        lookahead, stepsize, maxdims,
                        self.zi_, self.cov_,
                        return_states, self.method, weights, mem_splits)
    
    # remove lookahead portion from start
    print(X.size)
    X = X[:, lookahead_samples:] # Review this!!
    
    # # Return a modifier raw instance
    # raw = raw.copy()
    # raw.apply_function(lambda x: X, picks=picks,
    #                     channel_wise=False)
    return X


def asr_process(data, sfreq, state, windowlen=0.5, lookahead=0.25, stepsize=32,
                maxdims=0.66, Zi=None, cov=None,
                return_states=False, method="euclid", 
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
                R = np.eye(nc)  # trivial case
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