from __future__ import annotations  # For type hinting within class definitions

import abc
import datetime
import warnings
from collections.abc import Sequence
from types import NoneType
from typing import TYPE_CHECKING, Literal, Self, overload, override

import numpy as np
import numpy.typing as npt
from scipy import optimize as sopt

from pyNexafs.nexafs.normalisation.norm_settings import (
    # Config Classes
    configBase,
    configChannel,
    configEdges,
    configExternalChannel,
    edgeNormPost,
    edgeNormPre,
    extSelection,
    # Enumerates
    normMethod,
)
from pyNexafs.nexafs.scan import parsedScanAbstract, scanAbstract, scanBase
from pyNexafs.types import dtype
from pyNexafs.utils.functions import gaussian, lorentzian, pseudo_voigt

if TYPE_CHECKING:
    from pyNexafs.parsers import parserBase  # For type hinting only.


class scanAbstractNorm(scanAbstract, metaclass=abc.ABCMeta):
    """
    Abstract class to define the common methods used in all normalized scans.

    Parameters
    ----------
    scan : scan_abstract
        The original scan object to apply the normalisation to.
    """

    def __init__(
        self,
        scan: scanAbstract,
    ) -> None:
        self._origin = scan
        """The original scan object."""
        self._config = None
        """The normalisation configuration object, used to save settings."""

    @property
    def is_parsed(self) -> bool:
        """
        Whether the scan sources from a parser or not.

        Determines if `self.parser_class` can be accessed without raising a TypeError or ValueError.

        Returns
        -------
        bool
            True if the scan sources from a parser, False otherwise.
        """
        root_scan = self.root_scan
        if not isinstance(root_scan, scanBase):
            # Not a scanBase instance, cannot have a parser.
            return False
        else:
            if root_scan._parser_class is None:
                # Not loaded from a parser
                return False
            else:
                return True

    @property
    def parser(self) -> parserBase:
        """
        Recursively collect the original parser.

        This method traverses through any chained normalisation objects to find the root scan.

        Returns
        -------
        scan_abstract
            The original scan object.

        Raises
        ------
        ValueError
            If the root scan does not have a parser associated with it or has been dissociated from its parser.
        TypeError
            If the root scan is not a `scanBase` instance, which is required to access the parser.
        """
        root_scan = self.root_scan
        # Check if the root scan is a scanBase instance, which is required to access the parser.
        if not isinstance(root_scan, scanBase):
            raise TypeError(
                f"Root scan is a {type(root_scan)} instance (not a `scanBase`), and cannot access a linked parser."
            )
        if root_scan.parser is None:
            try:
                cls = root_scan.parser_class
                raise ValueError(
                    f"The root scan has been dissasociated from a parser (type `{cls.__name__}`)."
                )
            except ValueError:
                raise ValueError(
                    "The root scan has never been associated with a parser."
                )
        return root_scan.parser

    @property
    def parser_class(self):
        """
        Recursively collect the original parser.

        This method traverses through any chained normalisation objects to find the root scan.

        Returns
        -------
        scan_abstract
            The original scan object.

        Raises
        ------
        TypeError
            If the root scan is not a `scanBase` instance, which is required to access the parser.
        ValueError
            If the root scan has never been associated with a parser.
        """
        root_scan = self.root_scan
        # Check if the root scan is a scanBase instance, which is required to access the parser.
        if not isinstance(root_scan, scanBase):
            raise TypeError(
                f"Root scan is a {type(root_scan)} instance (not a `scanBase`), and cannot access a linked parser."
            )
        return root_scan.parser_class

    @property
    def root_scan(self) -> scanAbstract:
        """
        Recursively collect the original parser.

        This method traverses through any chained normalisation objects to find the root scan.

        Returns
        -------
        scan_abstract
            The original scan object.
        """
        if isinstance(self._origin, scanAbstractNorm):
            return self._origin.root_scan
        return self._origin

    @override
    @abc.abstractmethod
    def copy(self, newobj: Self | None = None) -> Self:
        """
        Copy the scan object.

        Parameters
        ----------
        newobj : Self | None, optional
            An optional instance of the same class to copy the data into.
            If None, a new instance is created using the constructor.
            This allows subclasses to control the copying process more finely,
            for example by avoiding calling the constructor again if not necessary.

        Returns
        -------
        scan_abstract
            A copy of the scan object.
        """
        if newobj is None:
            newobj = self.__class__(scan=self._origin)

        # Do not do a full copy, like the super method.
        # This is because properties like labels are linked, not copied.
        # Only copy the x,y,xerr,yerr data.
        newobj._x = self._x.copy() if self._x is not None else None
        newobj._y = self._y.copy() if self._y is not None else None
        newobj._x_errs = self._x_errs.copy() if self._x_errs is not None else None
        newobj._y_errs = self._y_errs.copy() if self._y_errs is not None else None
        return newobj

    @property
    def origin(self) -> scanAbstract:
        """
        Property for the original scan object.

        Returns
        -------
        scan_base
            The original scan object.
        """
        return self._origin

    @override
    def reload_labels_from_parser(self) -> None:
        return self._origin.reload_labels_from_parser()

    @property
    def ctime(self) -> datetime.datetime | None:
        """
        The creation time of the file as a datetime object.

        Returns
        -------
        datetime.datetime
            The creation time of the linked origin scan object.
        """
        origin = self._origin
        return origin.ctime if origin is not None else None

    @property
    def mtime(self) -> datetime.datetime | None:
        """
        The modification time of the file as a datetime object.

        Returns
        -------
        datetime.datetime
            The modification time of the linked origin scan object.
        """
        origin = self._origin
        return origin.mtime if origin is not None else None

    @property
    def filename(self) -> str | None:
        """
        Property for the filename of the scan object.

        Returns
        -------
        str | None
            The filename of the linked origin scan object.
        """
        origin = self._origin
        return origin.filename if isinstance(origin, scanBase) else None

    @abc.abstractmethod
    def _apply_normalisation(self) -> None:
        """
        Abstract method to apply the normalisation to the x/y data.
        """

    def _load_from_origin(self) -> None:
        """
        Method to reload data from the origin scan object.

        Uses instantiated scan reference.
        """
        # Copy data
        self._x = self._origin.x.copy() if self._origin.x is not None else None
        self._x_errs = (
            self._origin.x_errs.copy() if self._origin.x_errs is not None else None
        )
        self._y = self._origin.y.copy() if self._origin.y is not None else None
        self._y_errs = (
            self._origin.y_errs.copy() if self._origin.y_errs is not None else None
        )

    def load_and_normalise(self) -> None:
        """
        (Re)Load data from the origin and apply normalisation.

        Calls the `self._load_from_origin` method, which copies data from an original `scan` object,
        and then calls the `self._apply_normalisation` method, required to be implemented in subclasses.
        """
        self._load_from_origin()
        self._apply_normalisation()

    ## X property already covered by scan_abstract

    @scanAbstract.x_label.getter
    def x_label(self) -> str:
        """
        Property for the x-axis label, from the origin scan object.

        Returns
        -------
        str
            The x-axis label.
        """
        return self.origin.x_label

    @x_label.setter
    def x_label(self, label: str | None) -> None:
        """
        Property setter for the x-axis label, to the origin scan.

        Parameters
        ----------
        label : str
            The new x-axis label.
        """
        self.origin.x_label = label

    @scanAbstract.x_unit.getter
    def x_unit(self) -> str:
        """
        Property for the x-axis unit, from the origin scan object.

        Returns
        -------
        str
            The x-axis unit.
        """
        return self.origin.x_unit

    @x_unit.setter
    def x_unit(self, unit: str) -> None:
        """
        Property setter for the x-axis unit, to the origin scan.

        Parameters
        ----------
        unit : str
            The new x-axis unit.
        """
        self.origin.x_unit = unit

    @scanAbstract.y.setter
    def y(
        self, vals: npt.NDArray | list[list[int | float]] | list[int | float]
    ) -> None:
        """
        Property setter for the y data.

        Similar to @scan_abstract.y, but avoids resetting the `y_units`
        and `y_labels` attributes when setting the y data dimension, as
        labels should always be inherited from the original scan object.

        Parameters
        ----------
        vals : npt.NDArray
            The new y data.
        """
        # Set Y values
        if isinstance(vals, np.ndarray):
            if len(vals.shape) == 2:
                self._y = vals.copy()
            else:
                # Add additional axis to make y 2D instead of 1D
                self._y = vals.copy()[:, np.newaxis]
        elif isinstance(vals, list):
            if isinstance(vals[0], list):
                self._y = np.array(vals)
            else:
                # Assuming int/floats, add additional axis as well.
                self._y = np.array(vals)[:, np.newaxis]
        else:
            raise ValueError("Is not a list or a np.ndarray.")

        # Remove errs if number of yvals have changed.
        if self.y_errs is not None and self.y_errs.shape != self._y.shape:
            self.y_errs = None

        # Do not remove unit/labels if number of yvals have changed!

    @scanAbstract.y_labels.getter
    def y_labels(self) -> list[str]:
        """
        Property for the y-axis labels, from the origin scan object.

        Parameters
        ----------
        labels : list[str] | None
            The new y-axis labels, propagated to the origin scan object.

        Returns
        -------
        list[str]
            The y-axis labels.
        """
        return self.origin.y_labels

    @y_labels.setter
    def y_labels(self, labels: list[str] | None) -> None:
        self.origin.y_labels = labels

    @scanAbstract.y_units.getter
    def y_units(self) -> list[str]:
        """
        Property for the y-axis units, from the origin scan object.

        Parameters
        ----------
        units : list[str]
            The new y-axis units, propagated to the origin scan object.

        Returns
        -------
        list[str]
            The y-axis units.
        """
        return self.origin.y_units

    @y_units.setter
    def y_units(self, units: list[str] | None) -> None:
        self.origin.y_units = units

    @abc.abstractmethod
    def _config_class(self) -> type[configBase]:
        """
        The settings class for the normalisation configuration.

        Allows loading from a settings object.

        Returns
        -------
        type[configBase]
            The configuration class for the normalisation settings.
        """

    @property
    @abc.abstractmethod
    def settings(self) -> configBase:
        """
        The accumulated normalisation settings.

        Returns
        -------
        norm_config
            The normalisation settings.
        """

    @staticmethod
    @abc.abstractmethod
    def from_config(scan: scanAbstract, config: configBase) -> scanAbstractNorm:
        """
        Load the normalisation settings from a configuration object.

        Parameters
        ----------
        scan : scan_abstract
            The original scan object to apply the normalisation to.
        config : normConfig
            The normalisation configuration object.

        Returns
        -------
        scanAbstractNorm
            A new instance of a scanAbstractNorm subclass initialised with the provided configuration.
        """
        classes = scanAbstractNorm.__subclasses__()
        for cls in classes:
            if config.__class__ == cls._config_class:
                if cls.from_config != scanAbstractNorm.from_config:
                    new_obj = cls.from_config(scan, config)
                    return new_obj
                else:
                    raise NotImplementedError(
                        f"Class {cls} has not implemented the `from_config` method."
                    )
        raise TypeError(
            f"No matching class found for the configuration class {config.__class__}."
        )

    @property
    def channels(self) -> list[dtype]:
        """
        The list of channel labels.

        Returns
        -------
        list[str] | None
            The list of channel labels, from the origin scan object.
        """
        root = self.root_scan
        if isinstance(root, scanBase):
            return root.channels
        else:
            raise NotImplementedError(
                f"Root scan is {type(root).__name__}, not a scanBase instance, cannot get channels."
            )

    def label_index(self, label: str | dtype) -> int:
        """
        Get the index of a channel label.

        Parameters
        ----------
        label : str | dtype
            The channel label to get the index of.

        Returns
        -------
        int
            The index of the channel label.
        """
        # Use the base scan method to get the index of the label, which accounts for relabels as well.
        root = self.root_scan
        if isinstance(root, scanBase):
            return root.label_index(label)
        else:
            raise NotImplementedError(
                f"Root scan is {type(root).__name__}, not a scanBase instance, cannot get label index."
            )

    @override
    def __getitem__(self, key: str | dtype) -> npt.NDArray:
        """
        Get a specific channel of data by its dtype name.

        Parameters
        ----------
        key : str | dtype
            The name or dtype of the channel to retrieve.

        Returns
        -------
        npt.NDArray
            The data corresponding to the requested channel.
        """
        # Check if scan sources from a parser with relabelling or not.
        if self.is_parsed:
            # Use the parsedScanAbstract __getitem__ method, which accounts for relabels as well.
            return parsedScanAbstract.__getitem__(self, key)  # type: ignore
        else:
            # No relabels to use, default to super method.
            return super().__getitem__(key)


class scanAbstractYNorm(scanAbstractNorm, metaclass=abc.ABCMeta):
    """
    An abstract normalization class for applying normalisation to multiple y-channels.

    Parameters
    ----------
    scan : scan_abstract
        The original scan object to apply the normalisation to.
    apply_to : Sequence[str | dtype | int] | None, optional
        The list of y-channel labels (or indexes) to apply the normalisation to.
        Can also be a list of integers, representing the index of the y-channel, which
        will be converted to a list of strings to reduce ambiguity if labels are available.
        By default None (applies to all channels).
    conv_indexes : bool, optional
        Whether to convert integer indexes in `apply_to` to string labels if available.
        Default is True.
    """

    @override
    def __init__(
        self,
        scan: scanAbstract,
        apply_to: list[int | dtype | str] | None = None,
        conv_indexes: bool = True,
    ) -> None:
        # Set reference for original scan object
        super().__init__(scan)

        self._apply_to: Sequence[str | dtype | int] | None
        """The list of y-channel labels (or indexes) to apply the normalisation to."""

        # Set the apply_to attribute
        y_labels = scan.y_labels
        if apply_to is not None:
            if all(isinstance(i, str) for i in apply_to):
                # Create a copy
                self._apply_to = apply_to.copy()  # type: ignore - all i are str.
                return
            if y_labels is not None:
                if all(isinstance(i, (int, str, dtype)) for i in apply_to):
                    # Convert indexes to labels when available
                    self._apply_to = []
                    for i in range(len(apply_to)):
                        item = apply_to[i]
                        if conv_indexes and isinstance(item, int):
                            label = y_labels[item]
                            self._apply_to.append(
                                i if label is None else label
                            )  # Use the index
                        else:
                            # not conv_indexes and int, or string
                            self._apply_to.append(item)
                    return
                else:
                    raise ValueError(
                        "`apply_to` must be a list of integers or a list of strings."
                    )
            else:
                if all(isinstance(i, int) for i in apply_to):
                    # No labels available, use indexes
                    self._apply_to = apply_to.copy()  # type: ignore - all i are int.
                    return
                else:
                    raise ValueError(
                        "Y labels are not defined in the scan object. `apply_to` must be a list of integers, but some elements are strings."
                    )
        else:
            # Assign the apply_to attribute
            self._apply_to = None

    @override
    @abc.abstractmethod
    def copy(self) -> scanAbstractNorm:
        """
        Copy the scan object.

        Returns
        -------
        scan_abstract
            A copy of the scan object.
        """
        copy_obj = self.__class__(scan=self._origin, apply_to=self._apply_to)
        return copy_obj

    @property
    def apply_to(self) -> Sequence[str | dtype | int] | None:
        """
        The list of y-channel labels to apply the normalisation to.

        Parameters
        ----------
        apply_to : Sequence[str | dtype | int] | None
            The new list of y-channel labels (or indexes) to apply the normalisation to.
            If None, applies to all channels.

        Returns
        -------
        Sequence[str | dtype | int] | None
            The list of y-channel labels (or indexes) to apply the normalisation to.
            If None, applies to all channels.

        See Also
        --------
        scan_normalised.apply_to_indexes
            Method to return the indexes of the y-channels to apply the normalisation to.
        """
        return self._apply_to

    @apply_to.setter
    def apply_to(self, apply_to: Sequence[str | dtype | int] | None) -> None:
        self._apply_to = apply_to

    @overload
    def apply_to_indexes(
        self, with_labels: Literal[True]
    ) -> tuple[list[int], list[str | None]]: ...

    @overload
    def apply_to_indexes(self, with_labels: Literal[False]) -> list[int]: ...

    @overload
    def apply_to_indexes(self) -> list[int]: ...

    def apply_to_indexes(
        self, with_labels: bool = False
    ) -> list[int] | tuple[list[int], list[str | None]]:
        """
        The list of y-channel indexes to apply the normalisation to.

        Any labels not found in the y_labels are ignored, and a warning is raised.

        Parameters
        ----------
        with_labels : bool, optional
            If True, returns a tuple of a list of indexes and a list of labels, by default False.

        Returns
        -------
        list[int] | tuple[list[int], list[str | None]]
            If `with_labels` is False, returns a list of indexes.
            If `with_labels` is True, returns a tuple of indexes and labels.
            If no `apply_to` is defined, returns a list of all indexes.
        """
        at = self._apply_to
        ylabels = self.y_labels
        if at is not None:
            indexes: list[int] = []
            labels: list[str | None] = []
            ylabels = self.y_labels

            for label in at:
                if isinstance(label, int):
                    indexes.append(label)
                    labels.append(ylabels[label] if ylabels is not None else None)
                else:
                    try:
                        indexes.append(self.label_index(label))
                        labels.append(label)
                    except ValueError:
                        warnings.warn(f"Label {label} not found in y_labels. ")
            if with_labels:
                return indexes, labels
            return indexes
        else:
            if ylabels is not None:
                return list(range(len(ylabels)))
            elif self.y is not None:
                return list(range(self.y.shape[1]))
            else:
                raise ValueError(
                    "Y data is not defined, cannot determine number of channels to apply normalisation to."
                )


class scanEnergyNorm(scanAbstractNorm):
    """
    A normalization class for the X/energy axis of a scan, typically to a reference spectrum.

    Parameters
    ----------
    scan : scan_abstract
        The original scan object to apply the normalisation to.
    offset : float
        The relative (energy) shift to apply to the x data after normalisation.
    """

    @override
    def __init__(
        self,
        scan: scanAbstract,
        offset: float,
    ) -> None:
        # Set reference for original scan object
        super().__init__(scan)
        self._offset = offset
        """The relative (energy) shift to apply to the x data after normalisation."""

        # Process normalisation data
        self.load_and_normalise()

    @property
    def offset(self) -> float:
        """
        The relative (energy) shift to apply to the x data after normalisation.

        Parameters
        ----------
        offset : float
            The new relative (energy) shift to apply to the x data after normalisation.

        Returns
        -------
        float
            The relative (energy) shift to apply to the x data after normalisation.
        """
        return self._offset

    @offset.setter
    def offset(self, offset: float) -> None:
        self._offset = offset
        # Reapply normalisation
        self._apply_normalisation()

    @override
    def _apply_normalisation(self) -> None:
        if self._origin.x is not None:
            self._x = self._origin.x + self._offset

    @overload
    @staticmethod
    def calc_reference(
        reference_scan: scanAbstract,
        reference_channel: str | int | dtype = dtype.REF,
        *,
        fit_domain: tuple[float, float] | None = None,
        fit_fn: Literal["gaussian", "lorentzian", "voigt"] = "gaussian",
        p0: list[float] | None = None,
        full_output: Literal[False] = False,
    ) -> float: ...

    @overload
    @staticmethod
    def calc_reference(
        reference_scan: scanAbstract,
        reference_channel: str | int | dtype = dtype.REF,
        *,
        fit_domain: tuple[float, float] | None = None,
        fit_fn: Literal["gaussian", "lorentzian", "voigt"] = "gaussian",
        p0: list[float] | None = None,
        full_output: Literal[True],
    ) -> tuple[float, npt.NDArray, npt.NDArray]: ...

    @staticmethod
    def calc_reference(
        reference_scan: scanAbstract,
        reference_channel: str | int | dtype = dtype.REF,
        *,
        fit_domain: tuple[float, float] | None = None,
        fit_fn: Literal["gaussian", "lorentzian", "voigt"] = "gaussian",
        p0: list[float] | None = None,
        full_output: bool = False,
    ) -> float | tuple[float, npt.NDArray, npt.NDArray]:
        """
        Calculate the reference energy peak position for normalisation to a reference spectrum.

        Parameters
        ----------
        reference_scan : scanAbstract
            The reference scan to calculate the energy shift from.
        reference_channel : str | int | dtype, optional
            The y-channel name corresponding to the reference spectrum, by default dtype.REF.
        fit_domain : tuple[float, float] | None, optional
            The domain to perform the fit for energy shift calculation.
            If None, uses the full x range of the reference scan.
        fit_fn : Literal["gaussian", "lorentzian", "voigt"], optional
            The fitting function to use for calculating the energy shift, by default "gaussian".
        p0 : list[float] | None, optional
            The initial parameters for the fitting function, by default None,
            which uses automatic estimation based on the data.
        full_output : bool, optional
            Whether to return the full fit output (including fitted parameters and covariance), by default False.

        Returns
        -------
        float | tuple[float, npt.NDArray, npt.NDArray]
            If `full_output` is False,
                returns the calculated energy shift to apply to the x data for normalisation.
            If `full_output` is True,
                returns a tuple of the energy shift, fitted parameters, and covariance matrix from the fit.
        """
        x = reference_scan.x
        if x is None:
            raise ValueError(
                "X data is not defined in the reference scan, cannot calculate reference energy shift."
            )
        if isinstance(reference_channel, int):
            y = reference_scan.y
            if y is None:
                raise ValueError(
                    f"Y data is not defined in the reference scan, cannot get channel #{reference_channel} data."
                )
            y = y[:, reference_channel]
        else:
            y = reference_scan[reference_channel]

        # Get subset of data for fitting
        if fit_domain is not None:
            fit_idx = (x >= fit_domain[0]) & (x <= fit_domain[1])
            x_fit = x[fit_idx]
            y_fit = y[fit_idx]
        else:
            x_fit = x
            y_fit = y

        # Check there is enough data for fitting
        if len(x_fit) < 3:
            raise ValueError(
                f"Not enough data points in the fit domain ({fit_domain}) to perform fitting. "
                + f"At least 3 points are required ({len(x_fit)} found)."
            )

        # Perform the fit
        if fit_fn == "gaussian":
            fit_func = gaussian
            if p0 is None:
                imax = np.argmax(y_fit)
                p0 = [np.max(y_fit), x_fit[imax], (x_fit[-1] - x_fit[0]) / 10]
            assert len(p0) == 3, (
                "Initial parameters p0 for gaussian fit must be a list of 3 elements: [amplitude, center, width]."
            )
        elif fit_fn == "lorentzian":
            fit_func = lorentzian
            if p0 is None:
                imax = np.argmax(y_fit)
                p0 = [np.max(y_fit), x_fit[imax], (x_fit[-1] - x_fit[0]) / 10]
            assert len(p0) == 3, (
                "Initial parameters p0 for lorentzian fit must be a list of 3 elements: [amplitude, center, width]."
            )
        elif fit_fn == "voigt":
            fit_func = pseudo_voigt
            if p0 is None:
                imax = np.argmax(y_fit)
                p0 = [np.max(y_fit), x_fit[imax], (x_fit[-1] - x_fit[0]) / 10, 0.5]
            assert len(p0) == 4, (
                "Initial parameters p0 for voigt fit must be a list of 4 elements: [amplitude, center, width, shape]."
            )
        else:
            raise ValueError(
                f"Invalid fit function {fit_fn}. Must be one of 'gaussian', 'lorentzian', or 'voigt'."
            )

        try:
            popt, pcov = sopt.curve_fit(fit_func, x_fit, y_fit, p0=p0)
        except RuntimeError as e:
            raise RuntimeError(f"Fit did not converge: {e}")

        energy_peak_pos = popt[
            1
        ]  # The center parameter corresponds to the energy shift.

        if full_output:
            return energy_peak_pos, popt, pcov
        else:
            return energy_peak_pos

    @override
    def copy(self) -> Self:
        """
        Copy the scan object.

        Returns
        -------
        scanAbstractNorm
            A copy of the scan object.
        """
        copy_obj = self.__class__(scan=self._origin, offset=self._offset)
        return copy_obj

    @staticmethod
    def norm_to_reference(
        scan: scanAbstract,
        calibrated_peak_pos: float,
        channel: str | int | dtype = dtype.REF,
        *,
        fit_domain: tuple[float, float] | None = None,
        fit_fn: Literal["gaussian", "lorentzian", "voigt"] = "gaussian",
        reference: scanAbstract | None = None,
    ) -> scanEnergyNorm:
        """
        Normalise the x/energy axis of a scan to a reference spectrum.

        Parameters
        ----------
        scan : scanAbstract
            The original scan to normalise.
        calibrated_peak_pos : float
            The calibrated energy position of the reference peak to normalise to.
        channel : str | int | dtype, optional
            The y-channel name corresponding to the reference spectrum, by default dtype.REF.
        fit_domain : tuple[float, float] | None, optional
            The domain to perform the fit for energy shift calculation.
            If None, uses the full x range of the reference scan.
        fit_fn : Literal["gaussian", "lorentzian", "voigt"], optional
            The fitting function to use for calculating the energy shift, by default "gaussian".
        reference : scanAbstract | None, optional
            The reference scan to use for normalisation. If None, the input scan is used.

        Returns
        -------
        scanEnergyNorm
            The normalised scan object.
        """
        ref = reference if reference is not None else scan
        energy_peak_pos = scanEnergyNorm.calc_reference(
            reference_scan=ref,
            reference_channel=channel,
            fit_domain=fit_domain,
            fit_fn=fit_fn,
        )
        offset = calibrated_peak_pos - energy_peak_pos
        norm_scan = scanEnergyNorm(scan=scan, offset=offset)
        return norm_scan

    @override
    def _config_class(self) -> type[configBase]:
        """
        Return the settings class for the normalisation configuration.

        Required to allows loading from a settings object.

        Returns
        -------
        type[configChannel]
            The configuration class for the normalisation settings.
        """
        raise NotImplementedError(
            "Energy Calibration configuration class not yet implemented."
        )
        # return configX(self._offset)
        # TODO: Implement a Energy Calibration configuration class, which also takes a reference target value.

    @override
    @staticmethod
    def from_config(scan: scanAbstract, config: configBase) -> scanDoubleNorm:
        raise NotImplementedError(
            "Energy calibration from_config method not yet implemented."
        )

    @scanAbstractNorm.settings.getter
    @override
    def settings(self) -> configBase:
        raise NotImplementedError(
            "Energy calibration settings property not yet implemented."
        )


scanXNorm = (
    scanEnergyNorm  # Alias for scanEnergyNorm, as it is a normalisation of the X axis.
)


class scanNorm(scanAbstractYNorm):
    """
    General class for a normalisation of a scanAbstract.

    Normalisation is usually performed in reference to some flux measurement,
    such as the current of a mesh or photodiode. This class requires
    `norm_channel` or `norm_data` to be defined, but not both.

    Parameters
    ----------
    scan : Type[scan_base]
        The initial scan object to collect and normalise y dataseries.
    norm_channel : str | int | None, optional
        The y-channel name corresponding to normalisation data.
        Can provide a string name or an integer index, but will be stored as a string.
    norm_method : normMethod, optional
        The normalisation method to apply to the normalisation channel. See `configChannel.normMethod` for options.
    apply_to : Sequence[str | dtype | int] | None, optional
        The list of y-channel indexes to apply the normalisation to.
    conv_indexes : bool, optional
        Whether to convert integer indexes in `apply_to` to string labels if available.

    Raises
    ------
    ValueError
        If both `norm_channel` and `norm_data` are simultaneously `None` or not-`None`. Only one can be defined.
    ValueError
        If `norm_channel` is not found in y_labels.

    See Also
    --------
    norm_settings.configChannel.normMethod
        The normalisation method enumerate used for the normalisation channel.
    """

    normMethod = configChannel.normMethod

    @override
    def __init__(
        self,
        scan: scanAbstract,
        norm_channel: str | int | dtype,
        norm_method: normMethod = configChannel.normMethod.DIV,
        apply_to: Sequence[str | int | dtype] | None = None,
        conv_indexes: bool = True,
    ) -> None:
        # Set reference for original scan object
        super().__init__(scan, apply_to=apply_to, conv_indexes=conv_indexes)

        self._norm_idx: int
        """The index of the normalisation channel in the origin scan"""

        # Store initial information.
        ylabels = self._origin.y_labels
        if ylabels is None:
            raise ValueError(
                "Y labels are not defined in the scan object. Cannot match string label for `norm_channel`."
            )
        if isinstance(norm_channel, (str, dtype)):
            self._norm_idx = ylabels.index(norm_channel)
        else:
            self._norm_idx = norm_channel
            ylabel = ylabels[norm_channel]
            if ylabel is None:
                raise ValueError(
                    f"Y label at the specified index `{norm_channel}` is not defined. Cannot match string label for `norm_channel`."
                )
            norm_channel = ylabel

        # Save the normalisation channel settings as an object.
        self._config = configChannel(
            norm_method=norm_method, channel_name=norm_channel, apply_to=apply_to
        )
        """The normalisation configuration object used for serialising settings."""

        # Process normalisation data
        self.load_and_normalise()

    @override
    def _config_class(self) -> type[configBase]:
        """
        Return the settings class for the normalisation configuration.

        Required to allows loading from a settings object.

        Returns
        -------
        type[configChannel]
            The configuration class for the normalisation settings.
        """
        return configChannel

    @scanAbstractNorm.settings.getter
    @override
    def settings(self) -> configChannel:
        """
        A settings object capturing the settings for the normalisation channel.

        Returns
        -------
        configChannel
            The normalisation settings object, containing the channel and method.
        """
        return self._config

    @override
    @staticmethod
    def from_config(scan: scanAbstract, config: configChannel) -> scanNorm:  # type: ignore # TODO: implement fix for subtype override.
        """
        Load the normalisation settings from a configuration object.

        Parameters
        ----------
        scan : scan_abstract
            The original scan object to apply the normalisation to.
        config : configChannel
            The normalisation configuration object.

        Returns
        -------
        scanNorm
            A new instance of scanNorm initialised with the provided configuration.
        """
        return scanNorm(
            scan=scan, norm_channel=config.norm_channel, norm_method=config.method
        )

    @property
    def method(self) -> normMethod:
        """
        The normalisation method used for the normalisation channel.

        Parameters
        ----------
        method : normMethod
            The new normalisation method enumerate. Can be
            - 0: `normMethod.NONE` for no normalisation.
            - 1: `normMethod.SUB` for background subtraction.
            - 2: `normMethod.DIV` for normalisation by amplitude variation.

        Returns
        -------
        normMethod
            The normalisation method.
        """
        return self._config.method

    @method.setter
    def method(self, method: normMethod) -> None:
        self._config.method = method

    @property
    def norm_channel(self) -> str | None:
        """
        The normalisation channel name.

        Parameters
        ----------
        channel : str
            The new normalisation channel name.

        Returns
        -------
        str
            The normalisation channel name.
        """
        return self._config.norm_channel

    @norm_channel.setter
    def norm_channel(self, channel: str) -> None:
        self._config.norm_channel = channel

    @override
    def _load_from_origin(self) -> None:
        """
        Re-load from source, refreshing data, errors, labels and units for x,y variables.

        Overrides `scan_abstract_norm._load_from_origin` depending on the `_norm_channel` attribute being None.
        """
        # Y Reloading
        channel = self.norm_channel
        ind = self._norm_idx
        origin = self._origin
        if channel is not None:
            # Collect index of normalisation data
            if isinstance(origin, scanBase):
                # Use the base scan method to get the index of the label, which accounts for relabels as well.
                self._norm_idx = ind = origin.label_index(channel)
            else:
                y_labels = origin.y_labels
                if y_labels is not None:
                    self._norm_idx = ind = y_labels.index(channel)
                else:
                    raise ValueError(
                        "Y labels are not defined in the scan object. Cannot match string label for `norm_channel`."
                    )
        else:
            # idx is not None.
            y_labels = origin.y_labels
            if y_labels is not None:
                label = y_labels[ind]
                if label is not None:
                    self.norm_channel = label

        # Collect normalisaton data:
        self._norm_data = origin.y[:, ind] if origin.y is not None else None
        self._norm_data_errs = (
            origin.y_errs[:, ind] if origin.y_errs is not None else None
        )

        # Load x and y data regularly.
        super()._load_from_origin()

    @override
    def _apply_normalisation(self) -> None:
        """
        Perform the selected normalisation.

        `self.method` determines the method to apply to the provided y normalisation data.
        """
        match self.method:
            case self.normMethod.NONE:
                # Do nothing for no normalisation
                pass
            case self.normMethod.SUB:
                # Subtract the normalisation data from the y data.
                self._normalisation_subtract()
            case self.normMethod.DIV:
                # Scale the y data by the normalisation data.
                self._normalisation_divide()
            case self.normMethod.MULT:
                self._normalisation_multiply()
            case _:
                raise ValueError("Normalisation method not defined.")

    def _normalisation_multiply(self) -> None:
        # Multiply the y data by the normalisation data.
        ylabels = self.y_labels
        at = self._apply_to
        if (
            ylabels is None
            and at is not None
            and not all(isinstance(i, int) for i in at)
        ):
            raise ValueError(
                "Y labels are not defined in the scan object. Cannot match string labels in `apply_to`."
            )
        y = self._y
        if y is None:
            raise ValueError("Y data is not defined.")

        if at is None:
            for i in range(y.shape[1]):
                label = ylabels[i] if ylabels is not None else None
                if i == self._norm_idx:
                    # Skip normalisation channel
                    continue
                # Apply to all channels except the normalisation channel.
                y[:, i] *= self._norm_data
        else:
            for label in at:
                if isinstance(label, int):
                    i = label
                else:
                    i = self.label_index(label)
                if i == self._norm_idx:
                    raise ValueError(
                        f"Cannot apply normalisation to the normalisation channel itself (index {i}, label '{label}')."
                    )
                else:
                    y[:, i] *= self._norm_data

    def _normalisation_divide(self) -> None:
        """
        Scale the y data by the normalisation data.

        Applies the amplitude normalisation to `apply_to` channels, otherwise to all channels.
        """
        # Scale the y data, only in the apply_to channels.
        ylabels = self.y_labels
        at = self._apply_to
        if (
            ylabels is None
            and at is not None
            and not all(isinstance(i, int) for i in at)
        ):
            raise ValueError(
                "Y labels are not defined in the scan object. Cannot match string labels in `apply_to`."
            )
        y = self._y
        if y is None:
            raise ValueError("Y data is not defined.")

        if at is None:
            for i in range(y.shape[1]):
                label = ylabels[i] if ylabels is not None else None
                if i == self._norm_idx:
                    # Skip normalisation channel
                    continue
                # Apply to all channels except the normalisation channel.
                y[:, i] /= self._norm_data
        else:
            for label in at:
                if isinstance(label, int):
                    i = label
                else:
                    i = self.label_index(label)
                if i == self._norm_idx:
                    raise ValueError(
                        f"Cannot apply normalisation to the normalisation channel itself (index {i}, label '{label}')."
                    )
                else:
                    y[:, i] /= self._norm_data

    def _normalisation_subtract(self) -> None:
        """
        Subtract the background from the selected y data.

        Applies the background subtraction to `apply_to` channels, otherwise to all channels.
        """
        # Subtract the normalisation data from the y data.
        ylabels = self.y_labels
        at = self._apply_to
        if (
            ylabels is None
            and at is not None
            and not all(isinstance(i, int) for i in at)
        ):
            raise ValueError(
                "Y labels are not defined in the scan object. Cannot match string labels in `apply_to`."
            )
        y = self._y
        if y is None:
            raise ValueError("Y data is not defined.")

        if at is None:
            for i in range(y.shape[1]):
                label = ylabels[i] if ylabels is not None else None
                if i == self._norm_idx:
                    # Skip normalisation channel
                    continue
                # Apply to all channels except the normalisation channel.
                y[:, i] -= self._norm_data
        else:
            for label in at:
                if isinstance(label, int):
                    i = label
                else:
                    i = self.label_index(label)
                if i == self._norm_idx:
                    raise ValueError(
                        f"Cannot apply normalisation to the normalisation channel itself (index {i}, label '{label}')."
                    )
                else:
                    y[:, i] -= self._norm_data

    @scanAbstractNorm.y_labels.getter
    def y_labels(self) -> list[str | None] | None:
        """
        Property for the y-axis labels, from the origin scan object.

        Parameters
        ----------
        labels : list[str] | None
            The new y-axis labels.

        Returns
        -------
        list[str]
            The y-axis labels.
        """
        origin_labels = self._origin.y_labels
        ## No longer removing the normalisation channel from the labels, as it causes confusion when chaining normalisations.
        # idx = self._norm_idx
        # if idx is not None:
        #     return origin_labels[0:idx] + origin_labels[idx + 1 :]
        # else:
        return origin_labels

    @y_labels.setter
    def y_labels(self, labels: list[str] | None) -> None:
        idx = self._norm_idx
        if idx is not None and labels is not None and self.norm_channel is not None:
            # Include the normalisation channel in the labels when setting.
            self._origin.y_labels = labels[0:idx] + [self.norm_channel] + labels[idx:]
        else:
            # Otherwise set labels normally.
            self._origin.y_labels = labels

    @scanAbstractNorm.y_units.getter
    def y_units(self) -> list[str] | None:
        """
        Property for the y-axis units, from the origin scan object.

        When the normalisation channel is defined, the units are adjusted to remove the normalisation channel.

        Parameters
        ----------
        labels : list[str] | None
            The new y-axis units.

        Returns
        -------
        list[str] | None
            The y-axis units.
        """
        origin_units = self._origin.y_units
        idx = self._norm_idx
        if idx is not None and origin_units is not None:
            return origin_units[0:idx] + origin_units[idx + 1 :]
        else:
            return origin_units

    @y_units.setter
    def y_units(self, units: list[str] | None) -> None:
        idx = self._norm_idx
        if idx is not None:
            if units is not None:
                # Include the normalisation channel in the units when setting.
                self._origin.y_units = units[0:idx] + [self.norm_channel] + units[idx:]
                return
        # Otherwise, set units normally.
        self._origin.y_units = units
        return

    def copy(self) -> scanNorm:
        """
        A copy of the scan object.

        Returns
        -------
        scan_abstract
            A copy of the scan object.
        """
        return scanNorm(
            scan=self._origin,
            norm_channel=self.norm_channel,
            norm_method=self.method,
            apply_to=self._apply_to,
        )


class scanNormExt(scanNorm):
    """
    Normalising a scan_base to a particular channel of a background/external scan.

    Useful for NEXAFS such as carbon K edges. If a mesh current is used to normalise the signal,
    it will not necessarily normalise to the true absorption signal. A downstream photodiode, or alternative
    intensity channel, can be used to capture a true intensity signal. This channel can then
    permit double normalisation (i.e., with the mesh current signal) on the measured spectra.

    Parameters
    ----------
    scan : scanAbstract
        The initial scan object to collect and normalise.
    ext_scan : scanAbstract | str
        The background scan object to use for normalisation.
        Can also be a string filename or folder path.
    norm_channel : str
        The y-channel name corresponding to normalisation data in the background scan.
        This channel will be used to normalise the y data of the scan object.
    norm_method : normMethod, optional
        The normalisation method to apply to the normalisation channel. See `configChannel.normMethod` for options.
    apply_to : Sequence[str | dtype | int] | None, optional
        The list of y-channel labels to apply the normalisation to.
    conv_indexes : bool, optional
        Whether to convert integer indexes in `apply_to` to string labels if available.

    See Also
    --------
    norm_settings.normMethod
        The normalisation method enumerate used for the normalisation channel.
    norm_settings.extSelection
        The method to select an external scan from a folder path.

    Notes
    -----
    This class extends `scanNorm` to use a different scan object for normalisation data, but uses the same methods
    for normalisation (subtraction, division, multiplication).
    """

    normMethod = normMethod
    extSelection = extSelection

    @override
    def __init__(
        self,
        scan: scanAbstract,
        ext_scan: scanAbstract,
        norm_channel: str | int,
        norm_method: normMethod = configChannel.normMethod.DIV,
        apply_to: Sequence[str | dtype | int] | None = None,
        conv_indexes: bool = True,
    ) -> None:
        # Do not call super() for scanNorm, as we have a different configuration.
        # Set reference for original scan object
        scanAbstractYNorm.__init__(
            self, scan, apply_to=apply_to, conv_indexes=conv_indexes
        )

        # New variables
        self._norm_idx: int
        """The index of the normalisation channel in the background scan"""
        self._ext_scan: scanAbstract = ext_scan
        """The external/background scan object used for normalisation."""

        # Store initial information.
        ext_y_labels = ext_scan.y_labels
        if ext_y_labels is not None:
            if isinstance(norm_channel, str):
                self._norm_idx = ext_y_labels.index(norm_channel)
                channel = norm_channel
            else:
                self._norm_idx = norm_channel
                channel = ext_y_labels[norm_channel]
        else:
            raise ValueError(
                f"External scan has no y_labels defined. Cannot match normalisation channel name `{norm_channel}`."
            )

        # Redefine the normalisation index
        ext_ylabels = ext_scan.y_labels
        if ext_ylabels is None:
            raise ValueError(
                "Y labels are not defined in the external scan object. Cannot match string label for `norm_channel`."
            )
        self._norm_idx: int = (
            ext_ylabels.index(norm_channel)
            if isinstance(norm_channel, str)
            else norm_channel
        )

        # Save the normalisation channel settings as an object.
        self._config = configExternalChannel(
            channel_selection=extSelection.FIXED_SCAN,
            path=ext_scan.filepath,
            keyword=None,
            norm_method=norm_method,
            channel_name=channel,
            apply_to=apply_to,
        )
        """The normalisation configuration object used for serialising settings."""

        ### TODO: Make extra parameters into a configuration object that deals with folders rather than single files.
        # """
        # ext_selection : extSelection, optional
        #     The method to select the external scan when providing a folder path.
        #     Options are defined in `extSelection` enumerate.
        #     Defaults to `extSelection.FIXED_SCAN`, when a single filename is provided.
        # ext_keyword : str | None, optional
        #     The keyword to filter by, when selecting an external scan from a folder.
        # """
        # # If ext_scan is a string
        # path: str | None = None
        # if isinstance(ext_scan, str):
        #     if os.path.isfile(ext_scan):
        #         assert extSelection == extSelection.FIXED_SCAN, "When providing a single filename for ext_scan, ext_selection must be FIXED_SCAN."
        #         path = ext_scan
        #     elif os.path.isdir(ext_scan):
        #         assert extSelection != extSelection.FIXED_SCAN, "When providing a folder path for ext_scan, ext_selection cannot be FIXED_SCAN."
        #         path = ext_scan
        #     else:
        #         raise ValueError(f"Provided ext_scan string `{ext_scan}` is not a valid file or folder path.")
        # elif isinstance(ext_scan, scanAbstract):
        #     # Get the filename from the scan object
        #     path = ext_scan.filepath
        # else:
        #     raise TypeError(f"`ext_scan` must be a scanAbstract object or a string file/folder path. Got {type(ext_scan)}.")

        # Process normalisation data
        self.load_and_normalise()

    @override
    def _config_class(self) -> type[configBase]:
        """
        Return the settings class for the normalisation configuration.

        Required to allows loading from a settings object.

        Returns
        -------
        type[configChannel]
            The configuration class for the normalisation settings.
        """
        return configExternalChannel

    @scanNorm.norm_channel.getter
    def norm_channel(self) -> str | None:
        """
        The normalisation channel name from the background scan.

        Parameters
        ----------
        channel : str | None
            The new normalisation channel name.

        Returns
        -------
        str | None
            The normalisation channel name.
        """
        return self._config.norm_channel

    @norm_channel.setter
    def norm_channel(self, channel: str | None) -> None:
        self._config.norm_channel = channel

    @norm_channel.deleter
    def norm_channel(self) -> None:
        self._config.norm_channel = None

    @override
    def _load_from_origin(self) -> None:
        """
        Re-load from source, refreshing data, errors, labels and units for x,y variables.

        Overrides `scanNorm._load_from_origin` depending on the `_norm_channel` attribute being None.
        """
        # Y Reloading
        channel = self.norm_channel
        ext_scan = self._ext_scan
        if ext_scan is not None:
            ext_ylabels = ext_scan.y_labels
            # Update using the channel name
            if ext_ylabels is not None:
                if channel is not None:
                    self._norm_idx = ind = ext_ylabels.index(channel)
                    self.norm_channel = channel
                else:
                    self.norm_channel = ext_ylabels[self._norm_idx]
                    ind = self._norm_idx
            else:
                raise ValueError("External scan y_labels are not defined.")

            # Collect normalisaton data:
            ext_y = ext_scan.y
            if ext_y is None:
                raise ValueError("External scan y data is not defined.")
            self._norm_data = ext_y[:, ind].copy() if ext_y is not None else None
            self._norm_data_errs = (
                self._ext_scan.y_errs[:, ind].copy()
                if self._ext_scan.y_errs is not None
                else None
            )
        else:
            raise ValueError("External scan is not defined.")

        # Load x and y data regularly; skip the `scanNorm._load_from_origin` to avoid using `self._norm_idx`
        scanAbstractYNorm._load_from_origin(self)


class scanNormEdges(scanAbstractYNorm):
    """
    Normalising a scan_base across pre &/ post edges.

    Uses two band definitions to normalise the y data centred around a chemical resonant edge
    such as the K-edge of carbon.

    Parameters
    ----------
    scan : Type[scan_base] | scan_base
        The initial scan object to collect and normalise y dataseries.
    pre_edge_domain : _type_, optional
        Data to define the domain of pre-edge normalisation. Can take the following data forms:
        - A list of indices (integers) to subset the y data.
        - A tuple of floats to define the inclusive-domain of the x data.
        - None, to not perform pre-edge normalisation. Overrides pre_edge_normalisation enumerate.
    post_edge_domain : _type_, optional
        Data to define the domain of post-edge normalisation. Same format as pre-edge.
        If None overrides post_edge_normalisation enumerate.
    pre_edge_norm_method : edgeNormPre, optional
        Normalisation type for pre-edge, by default edgeNormPre.LINEAR.
    post_edge_norm_method : EDGE_NORM_TYPE, optional
        Normalisation type for post-edge, by default EDGE_NORM_TYPE.LINEAR.
    pre_edge_level : float, optional
        Normalisation level for pre-edge, by default 0.1.
    post_edge_level : float, optional
        Normalisation level for post-edge, by default 1.0.
    apply_to : Sequence[str | dtype | int] | None, optional
        The sequence of y-channel labels to apply the normalisation to.

    Attributes
    ----------
    edgeNormPre : enumerate
        Enumerate types for pre edge normalisation.
    edgeNormPost : enumerate
        Enumerate types for post edge normalisation.

    Raises
    ------
    ValueError
        If both `norm_channel` and `norm_data` are simultaneously None or not-None. Only one can be defined.
    ValueError
        If `norm_channel` is not found in y_labels.
    """

    @staticmethod
    def LIN_FN(x: npt.NDArray, m: float, c: float) -> npt.NDArray:
        return m * x + c

    """
    A function to define a linear fit.

    Parameters
    ----------
    x : ArrayLike
        The x value.
    m : float
        The gradient of the line.
    c : float
        The y-intercept of the line.
    """

    @staticmethod
    def EXP_FN_OFFSET(x: npt.NDArray, a: float, b: float, c: float) -> npt.NDArray:
        return a * np.exp(b * (x - x.min())) + c

    """
    A function to define an exponential fit, adjusted to the minimum x value.

    Parameters
    ----------
    x : ArrayLike
        The x value.
    a : float
        The amplitude of the exponential.
    b : float
        The decay constant of the exponential.
    c : float
        The offset of the exponential.
    """

    edgeNormPre = edgeNormPre
    edgeNormPost = edgeNormPost

    def __init__(
        self,
        scan: scanAbstract,
        pre_edge_domain: tuple[float, float] | list[int] | None,
        post_edge_domain: tuple[float, float] | list[int] | None,
        pre_edge_norm_method: edgeNormPre = edgeNormPre.CONSTANT,
        post_edge_norm_method: edgeNormPost = edgeNormPost.CONSTANT,
        pre_edge_level: float | None = None,
        post_edge_level: float | None = None,
        apply_to: Sequence[str | dtype | int] | None = None,
    ) -> None:
        # Initialize the scan_abstract_norm class.
        super().__init__(scan=scan, apply_to=apply_to)
        # Check level defaults
        if pre_edge_level is None:
            pre_edge_level = configEdges.DEFAULT_PRE_EDGE_LEVEL_CONSTANT
        if post_edge_level is None:
            post_edge_level = configEdges.DEFAULT_POST_EDGE_LEVEL_CONSTANT
        # Create a settings object to store the normalisation settings.
        self._config = configEdges(
            pre_edge_domain=pre_edge_domain,
            post_edge_domain=post_edge_domain,
            pre_edge_norm_method=pre_edge_norm_method,
            post_edge_norm_method=post_edge_norm_method,
            pre_edge_level=pre_edge_level,
            post_edge_level=post_edge_level,
            apply_to=apply_to,
        )

        # Define variables for edge fitting
        self.pre_edge_fit_params: list | None = None
        self.post_edge_fit_params: list | None = None

        # Perform normalisation
        self.load_and_normalise()

    @override
    def _config_class(self) -> type[configBase]:
        """
        The linked configuration class for the normalisation settings.

        Returns
        -------
        type[configEdges]
            The configuration class for the normalisation settings.
        """
        return configEdges

    @scanAbstractNorm.settings.getter
    @override
    def settings(self) -> configEdges:
        """
        A settings object associated with the settings for a normalisation channel.

        Returns
        -------
        configEdges
            The normalisation settings.
        """
        return self._config

    @override
    @staticmethod
    def from_config(scan: scanAbstract, config: configEdges) -> scanNormEdges:  # type: ignore # TODO: implement fix for subtype override.
        """
        Load the normalisation settings from a configuration object.

        Parameters
        ----------
        scan : scan_abstract
            The original scan object to apply the normalisation to.
        config : configEdges
            The normalisation configuration object.

        Returns
        -------
        scanNormEdges
            A new instance of scanNormEdges with the provided configuration.
        """
        return scanNormEdges(
            scan=scan,
            pre_edge_domain=config.pre_edge_domain,
            post_edge_domain=config.post_edge_domain,
            pre_edge_norm_method=config.pre_edge_norm_method,
            post_edge_norm_method=config.post_edge_norm_method,
            pre_edge_level=config.pre_edge_level,
            post_edge_level=config.post_edge_level,
            apply_to=config.apply_to,
        )

    @override
    def copy(self) -> scanNormEdges:
        clone = scanNormEdges(
            self._origin,
            pre_edge_domain=self._config._pre_edge_domain,
            post_edge_domain=self._config._post_edge_domain,
            pre_edge_norm_method=self._config._pre_edge_norm_method,
            post_edge_norm_method=self._config._post_edge_norm_method,
            pre_edge_level=self._config._pre_edge_level,
            post_edge_level=self._config._post_edge_level,
            apply_to=self._config._apply_to,
        )
        return clone

    @property
    def DEFAULT_PRE_EDGE_LEVEL_CONSTANT(self) -> float:
        return self._config.DEFAULT_PRE_EDGE_LEVEL_CONSTANT

    @property
    def DEFAULT_PRE_EDGE_LEVEL_LINEAR(self) -> float:
        return self._config.DEFAULT_PRE_EDGE_LEVEL_LINEAR

    @property
    def DEFAULT_PRE_EDGE_LEVEL_EXP(self) -> float:
        return self._config.DEFAULT_PRE_EDGE_LEVEL_EXP

    @property
    def DEFAULT_POST_EDGE_LEVEL_CONSTANT(self) -> float:
        return self._config.DEFAULT_POST_EDGE_LEVEL_CONSTANT

    @override
    def _apply_normalisation(self) -> None:
        """
        Scale y data by the normalisation regions.

        Applies the pre-edge and post-edge normalisation methods to the y data.
        """
        # Define fitting functions
        lin_fn = self.LIN_FN
        exp_fn = self.EXP_FN_OFFSET
        # Perform dual normalisation, sequentially to avoid duplicate code.

        ## PRE-EDGE
        if (
            self.pre_edge_norm_method is not edgeNormPre.NONE
            and self.pre_edge_domain is not None
        ):
            # Collect pre-edge indexes
            if isinstance(self.pre_edge_domain, list):
                # Use the list of indexes
                pre_inds = self.pre_edge_domain
                if len(pre_inds) == 0:
                    raise ValueError("Pre-edge index list is empty.")
            elif isinstance(self.pre_edge_domain, tuple):
                # Use the domain range
                pre_inds = np.asarray(
                    (self.x >= self.pre_edge_domain[0])
                    & (self.x <= self.pre_edge_domain[1])
                ).nonzero()
                # Check indexes of each tuple element
                if len(pre_inds[0]) == 0:
                    raise ValueError(
                        f"Pre-edge domain ({self.pre_edge_domain[0]} to {self.pre_edge_domain[1]}) contains no datapoints."
                    )
            else:
                raise AttributeError(
                    "Pre-edge domain is not defined correctly. Should be a list of indexes or the range in a tuple."
                )

            # Calculate pre-edge on all y values and normalise
            match self.pre_edge_norm_method:
                case edgeNormPre.CONSTANT:
                    inds = (
                        self.apply_to_indexes()
                        if self.apply_to
                        else np.arange(self.y.shape[1])
                    )
                    mean = np.mean(self.y[pre_inds][:, inds], axis=0)
                    stdev = np.std(self.y[pre_inds][:, inds], axis=0)
                    self.y[:, inds] += -mean[np.newaxis, :] + self.pre_edge_level
                    self.pre_edge_fit_params = mean.tolist()

                    if self.y_errs is not None:
                        # Add the standard deviation
                        self.y_errs[:, inds] = np.sqrt(
                            np.square(self.y_errs[:, inds]) + stdev[np.newaxis, :] ** 2
                        )

                case edgeNormPre.LINEAR:
                    params: list = []
                    inds = (
                        self.apply_to_indexes()
                        if self.apply_to
                        else np.arange(self.y.shape[1])
                    )
                    for i in range(self.y.shape[1]):
                        if i in inds:
                            popt, pcov = sopt.curve_fit(
                                lin_fn, self.x[pre_inds], self.y[:, i][pre_inds]
                            )
                            self.pre_edge_fit_params = popt.tolist()
                            self.y[:, i] += -lin_fn(self.x, *popt) + self.pre_edge_level
                            params += popt.tolist()

                    self.pre_edge_fit_params = params

                case edgeNormPre.EXPONENTIAL:
                    # TODO: Implement apply to
                    params: list = []
                    inds = (
                        self.apply_to_indexes()
                        if self.apply_to
                        else np.arange(self.y.shape[1])
                    )
                    for i in range(self.y.shape[1]):
                        if i in inds:
                            # Subtract offset to fit
                            xfit = self.x[pre_inds] - self.x[pre_inds].min()
                            x = self.x - self.x[pre_inds].min()
                            a0 = (
                                self.y[:, i][pre_inds].max()
                                - self.y[:, i][pre_inds].min()
                            )
                        # Fit
                        popt, pcov = sopt.curve_fit(
                            exp_fn,
                            xfit,
                            self.y[:, i][pre_inds],
                            p0=(a0, -0.5, 0),
                            bounds=[(-np.inf, -np.inf, -np.inf), (np.inf, 0, np.inf)],
                            maxfev=10000,
                        )
                        self.y[:, i] += -exp_fn(x, *popt) + self.pre_edge_level
                        params += popt.tolist()

                    self.pre_edge_fit_params = params

                case edgeNormPre.NONE:
                    # Do nothing for no pre-edge normalisation
                    pass
                case _:
                    # Should never reach here, and dual normalisation excludes None type.
                    raise ValueError("Pre-edge normalisation type not defined.")

        ## POST-EDGE
        if (
            self.post_edge_norm_method is not edgeNormPost.NONE
            and self.post_edge_domain is not None
        ):
            # Collect post-edge indexes:
            if isinstance(self.post_edge_domain, list):
                post_inds = self.post_edge_domain
                if len(post_inds) == 0:
                    raise ValueError("Post-edge index list is empty.")
            elif isinstance(self.post_edge_domain, tuple):
                post_inds = np.where(
                    (self.x >= self.post_edge_domain[0])
                    & (self.x <= self.post_edge_domain[1])
                )
                # Check indexes of each tuple element
                if len(post_inds[0]) == 0:
                    raise ValueError(
                        f"Post-edge domain ({self.post_edge_domain[0]} to {self.post_edge_domain[1]}) contains no datapoints."
                    )
            else:
                raise AttributeError(
                    "Post-edge domain is not defined correctly. Should be a list of indexes or the range in a tuple."
                )

            if len(post_inds) == 0:
                raise ValueError("Post-edge domain is empty.")

            # Calculate post-edge and normalise. Note, y data is already normalised by pre-edge.
            match self.post_edge_norm_method:
                case edgeNormPost.CONSTANT:
                    inds = (
                        self.apply_to_indexes()
                        if self.apply_to
                        else np.arange(self.y.shape[1])
                    )

                    if self.apply_to:
                        inds = self.apply_to_indexes()
                        postave = np.mean(self.y[post_inds][:, inds], axis=0)
                        scale = np.zeros(postave.shape)
                        scale[postave != 0] = (
                            self.post_edge_level - self.pre_edge_level
                        ) / postave[postave != 0]
                        self.y[:, inds] = (
                            self.y[:, inds] - self.pre_edge_level
                        ) * scale[np.newaxis, :] + self.pre_edge_level
                    else:
                        # Get the mean of the post-edge from the pre-edge level.
                        postave = np.mean(
                            self.y[post_inds] - self.pre_edge_level, axis=0
                        )
                        # Scale to difference from pre-edge level to post-edge level. Ignore zero values.
                        scale = np.zeros(postave.shape)
                        scale[postave != 0] = (
                            self.post_edge_level - self.pre_edge_level
                        ) / postave[postave != 0]
                        self.y = (
                            self.y - self.pre_edge_level
                        ) * scale + self.pre_edge_level
                case edgeNormPost.NONE:
                    # Do nothing
                    pass
                case _:
                    # Should never reach here, and dual normalisation excludes None type.
                    raise ValueError(
                        f"Post-edge normalisation type not defined for {self.post_edge_norm_method}."
                    )

    @property
    def pre_edge_domain(self) -> tuple[float, float] | list[int] | None:
        """
        A property defining the pre-edge domain of normalisation.

        If setting and `pre_edge_normalisation` is `NONE`, will set new normalisation enumerate to `LINEAR`.

        Parameters
        ----------
        vals : list[int] | tuple[float, float] | None
            Can be defined using a list of included indices matching x datapoints.
            Alternatively a tuple of two floats defining the inclusive endpoints.
            Alternatively None stops normalisation being performed on the pre-edge.

        Returns
        -------
        list[int] | tuple[float, float] | None
            Same as vals parameter.
        """
        return self._config._pre_edge_domain

    @pre_edge_domain.setter
    def pre_edge_domain(self, vals: list[int] | tuple[float, float] | None) -> None:
        self._config._pre_edge_domain = vals

    @pre_edge_domain.deleter
    def pre_edge_domain(self):
        del self._config._pre_edge_domain

    @property
    def pre_edge_norm_method(self) -> edgeNormPre:
        """
        Property to define the type of normalisation performed on the pre-edge.

        Parameters
        ----------
        vals : edgeNormPre
            LINEAR, EXPONENTIAL or NONE.

        Returns
        -------
        edgeNormPre
            The current normalisation type.
        """
        return self._config.pre_edge_norm_method

    @pre_edge_norm_method.setter
    def pre_edge_norm_method(self, vals: edgeNormPre) -> None:
        self._config.pre_edge_norm_method = vals

    @pre_edge_norm_method.deleter
    def pre_edge_norm_method(self) -> None:
        del self._config.pre_edge_norm_method

    @property
    def pre_edge_level(self) -> float:
        """
        Property to define the normalisation level (constant) for the pre-edge.

        Parameters
        ----------
        vals : float
            The normalisation average.

        Returns
        -------
        float
            The normalisation level.
        """
        return self._config.pre_edge_level

    @pre_edge_level.setter
    def pre_edge_level(self, level: float) -> None:
        self._config.pre_edge_level = level

    @pre_edge_level.deleter
    def pre_edge_level(self) -> None:
        del self._config.pre_edge_level

    @property
    def post_edge_domain(self) -> list[int] | tuple[float, float] | None:
        """
        A property defining the post-edge domain of normalisation.

        If setting and `post_edge_normalisation` is `NONE`, will set new normalisation enumerate to `LINEAR`.

        Parameters
        ----------
        vals : list[int] | tuple[float, float] | None
            Can be defined using a list of included indices matching x datapoints.
            Alternatively a tuple of two floats defining the inclusive endpoints.
            Alternatively None stops normalisation being performed on the post-edge.

        Returns
        -------
        list[int] | tuple[float, float] | None
            Same as vals parameter.
        """
        return self._config._post_edge_domain

    @post_edge_domain.setter
    def post_edge_domain(self, vals: list[int] | tuple[float, float] | None) -> None:
        self._config.post_edge_domain = vals

    @post_edge_domain.deleter
    def post_edge_domain(self):
        del self._config.post_edge_domain

    @property
    def post_edge_norm_method(self) -> edgeNormPost:
        """
        Property to define the type of normalisation performed on the post-edge.

        Parameters
        ----------
        vals : edgeNormPost
            LINEAR, EXPONENTIAL or NONE.

        Returns
        -------
        edgeNormPost
            The current normalisation type.
        """
        return self._config._post_edge_norm_method

    @post_edge_norm_method.setter
    def post_edge_norm_method(self, method: edgeNormPost) -> None:
        self._config.post_edge_norm_method = method

    @post_edge_norm_method.deleter
    def post_edge_norm_method(self) -> None:
        self._post_edge_normalisation = None

    @property
    def post_edge_level(self) -> float:
        """
        Property to define the normalisation level (constant) for the post-edge.

        Parameters
        ----------
        vals : float
            The normalisation average.

        Returns
        -------
        float
            The normalisation level.
        """
        return self._config.post_edge_level

    @post_edge_level.setter
    def post_edge_level(self, vals: float) -> None:
        # if (
        #     self.post_edge_normalisation
        #     is edgeNormPost.EXPONENTIAL
        # ):
        #     if vals <= 0:
        #         raise ValueError(
        #             "Exponential normalisation requires a positive, non-zero level."
        #         )
        self._config.post_edge_level = vals

    @post_edge_level.deleter
    def post_edge_level(self) -> None:
        if (
            # self.post_edge_normalisation
            # is edgeNormPost.EXPONENTIAL
            False
        ):
            self._post_edge_level = configEdges.DEFAULT_PRE_EDGE_LEVEL_EXP
        # elif self.pre_edge_normalisation is configEdges.EDGE_NORM_TYPE.LINEAR:
        # self._pre_edge_level = configEdges.DEFAULT_PRE_EDGE_LEVEL_LINEAR
        else:
            self._post_edge_level = configEdges.DEFAULT_PRE_EDGE_LEVEL_LINEAR


class scanDoubleNorm(scanAbstractYNorm):
    r"""
    A scan object normalised by the combination of an internal and external channel.

    The normalisation is calculated by dividing the y data by the internal normalisation channel,
    then multiplying by the same normalisation channel from an external scan object, before dividing
    by a second normalisation channel from the external scan object (i.e. a direct photodiode).

    .. math::
        y_{norm} = \frac{y_{scan}}{y_{norm\_channel}} \times \frac{y_{ext\_norm\_channel}}{y_{ext\_double\_norm\_channel}}

    Parameters
    ----------
    scan : scanAbstract
        The initial scan object to collect and normalise.
    ext_scan : scanAbstract
        The external scan object to use for normalisation.
    norm_channel : str | int
        The y-channel name corresponding to normalisation data in the original scan.
    double_norm_channel : str | int
        The y-channel name corresponding to normalisation data in the external scan.
    apply_to : Sequence[str | dtype | int] | None, optional
        The sequence of y-channel labels to apply the normalisation to.
    conv_indexes : bool, optional
        Whether to convert indexes in apply_to to labels.
    """

    def __init__(
        self,
        scan: scanAbstract,
        ext_scan: scanAbstract,
        norm_channel: str | int,
        double_norm_channel: str | int,
        apply_to: Sequence[str | dtype | int] | None = None,
        conv_indexes: bool = True,
    ) -> None:
        ext_ylabels = ext_scan.y_labels
        ylabels = scan.y_labels
        if isinstance(norm_channel, int) or isinstance(double_norm_channel, int):
            assert ylabels == ext_ylabels, (
                "When providing integer indexes for both normalisation channels, the y_labels of both scans must \
                match to ensure normalisation is being applied correctly."
            )

        # Automatically remove the channels from the apply_to list.
        if isinstance(double_norm_channel, int):
            double_norm_channel_idx = double_norm_channel
        else:
            if ext_ylabels is None:
                raise ValueError("External scan y_labels are not defined.")
            else:
                double_norm_channel_idx = ext_ylabels.index(double_norm_channel)

        if isinstance(norm_channel, int):
            norm_channel_idx = norm_channel
        else:
            if ylabels is None:
                raise ValueError("Scan y_labels are not defined.")
            else:
                norm_channel_idx = ylabels.index(norm_channel)

        if apply_to is not None:
            apply_to = [
                a
                for a in apply_to
                # Avoid all four possibilities
                if a != norm_channel
                and a != double_norm_channel
                and a != norm_channel_idx
                and a != double_norm_channel_idx
            ]
        else:
            apply_to = [
                a
                for a in (list(range(len(ylabels))) if ylabels is not None else [])
                # Avoid both index possibilities
                if a != norm_channel_idx and a != double_norm_channel_idx
            ]

        # Initialize the scan_abstract_norm class.
        super().__init__(scan=scan, apply_to=apply_to, conv_indexes=conv_indexes)

        # Save the reference to the norm scan
        self._ext_scan: scanAbstract = ext_scan
        """The external/background scan object used for normalisation."""
        self._orig_norm_channel: str | int = norm_channel
        """The original normalisation channel from the scan object."""
        self._orig_double_norm_channel: str | int = double_norm_channel
        """The original double normalisation channel from the external scan object."""

        # Use the existing scanNorm and scanNormExt classes to perform the normalisation.
        self._norm1 = scanNorm(
            scan=scan,
            norm_channel=norm_channel,
            norm_method=configChannel.normMethod.DIV,
            apply_to=apply_to,
        )
        self._norm2 = scanNormExt(
            scan=self._norm1,
            ext_scan=ext_scan,
            norm_channel=norm_channel,
            norm_method=configChannel.normMethod.MULT,
            apply_to=apply_to,
        )
        self._norm3 = scanNormExt(
            scan=self._norm2,
            ext_scan=ext_scan,
            norm_channel=double_norm_channel,
            norm_method=configChannel.normMethod.DIV,
            apply_to=apply_to,
        )
        # Copy the pointers to the final normalised data to this object.
        self._x = self._norm3.x
        self._x_errs = self._norm3.x_errs
        self._y = self._norm3.y
        self._y_errs = self._norm3.y_errs

        # Perform normalisation
        # self.load_and_normalise()

    @property
    def ext_scan(self) -> scanAbstract:
        """
        The external scan object used for normalisation.

        Returns
        -------
        scanAbstract
            The external scan object.
        """
        return self._ext_scan

    @ext_scan.setter
    def ext_scan(self, scan: scanAbstract) -> None:
        self._ext_scan = scan
        # Propogate to normalisation objects
        self._norm2._ext_scan = scan
        self._norm3._ext_scan = scan

    @override
    def load_and_normalise(self) -> NoneType:
        # Need to sequentially load and normalise each step.
        self._norm1.load_and_normalise()
        self._norm2.load_and_normalise()
        self._norm3.load_and_normalise()
        # Copy the pointers to the final normalised data to this object.
        self._x = self._norm3.x
        self._x_errs = self._norm3.x_errs
        self._y = self._norm3.y
        self._y_errs = self._norm3.y_errs

    @override
    def _load_from_origin(self) -> NoneType:
        # Call the three normalisation steps to refresh data.
        raise NotImplementedError(
            "Double normalisation `_load_from_origin` method will not function as intended, \
                                  as 3 normalisation steps require sequential load and normalisation application."
        )

    @override
    def _apply_normalisation(self) -> None:
        raise NotImplementedError(
            "Double normalisation `_apply_normalisation` method will not function as intended, \
                                  as 3 normalisation steps require sequential load and normalisation application."
        )

    @override
    def _config_class(self) -> type[configBase]:
        """
        Return the settings class for the normalisation configuration.

        Required to allows loading from a settings object.

        Returns
        -------
        type[configChannel]
            The configuration class for the normalisation settings.
        """
        raise NotImplementedError(
            "Double normalisation configuration class not yet implemented."
        )
        return configBase  # TODO: Implement a double normalisation configuration class.

    @override
    @staticmethod
    def from_config(scan: scanAbstract, config: configBase) -> scanDoubleNorm:
        raise NotImplementedError(
            "Double normalisation from_config method not yet implemented."
        )

    @scanAbstractNorm.settings.getter
    @override
    def settings(self) -> configBase:
        raise NotImplementedError(
            "Double normalisation settings property not yet implemented."
        )

    @override
    def copy(self, *args, **kwargs) -> Self:
        clone = self.__class__(
            scan=self._origin,
            ext_scan=self._norm2._ext_scan,
            norm_channel=self._orig_norm_channel,
            double_norm_channel=self._orig_double_norm_channel,
            apply_to=self._apply_to,
            *args,
            **kwargs,
        )
        return clone


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt

    import pyNexafs

    # Create a basic scan object form test data
    MEX2_MDA_PATH = os.path.abspath(
        os.path.normpath(
            os.path.join(
                # os.path.dirname(__file__),
                os.path.dirname(pyNexafs.__file__),
                "..\\tests\\test_data\\au\\MEX2\\2024-03\\MEX2_5640.mda",
            )
            .replace("\\", os.sep)
            .replace("/", os.sep)
        )
    )
    SXR_MDA_PATH = (
        os.path.abspath(
            os.path.normpath(
                os.path.join(
                    # Use package root
                    os.path.dirname(pyNexafs.__file__),
                    "..\\tests\\test_data\\au\\SXR\\2024-03\\sxr129598.mda",
                )
            )
        )
        .replace("\\", os.sep)
        .replace("/", os.sep)
    )
    from pyNexafs.parsers.au import MEX2_NEXAFS, SXR_NEXAFS

    # MEX2 example
    parser_mex2 = MEX2_NEXAFS(
        MEX2_MDA_PATH, relabel=True, energy_bin_domain=(3300, 3700)
    )
    scan_mex2 = parser_mex2.to_scan()

    # SXR Oxygen example
    parser_sxr = SXR_NEXAFS(SXR_MDA_PATH, relabel=True)
    scan_sxr = parser_sxr.to_scan()
    assert scan_sxr.y_labels[2] == "I0 VF", scan_sxr.y_labels[2]

    examples = [
        {
            "scan": scan_mex2,
            "norm_channel": "I0",
            "pre_edge_domain": (2460, 2465),
            "post_edge_domain": (2520, 2530),
            "idx": 1,
        },
        {
            "scan": scan_sxr,
            "norm_channel": scan_sxr.y_labels[2],
            "pre_edge_domain": (515, 525),
            "post_edge_domain": (570, 580),
            "idx": 0,
        },
    ]

    # plt.subplots(1,2, figsize=(10,5))
    # plt.plot(scan_sxr.x, scan_sxr.y[:,2], label="I0")
    # plt.show()

    for ex in examples[1:]:
        scan = ex["scan"]
        norm_channel = ex["norm_channel"]
        pre_edge_domain = ex["pre_edge_domain"]
        post_edge_domain = ex["post_edge_domain"]
        idx = ex["idx"]

        # Normalise to a channel
        norm = scanNorm(scan, norm_channel=norm_channel)

        # Normalise to the edges
        edge_c = configEdges(
            scan, pre_edge_domain=pre_edge_domain, post_edge_domain=post_edge_domain
        )
        edge_l = configEdges(
            scan,
            pre_edge_domain=pre_edge_domain,
            post_edge_domain=post_edge_domain,
            pre_edge_normalisation=edgeNormPre.LINEAR,
        )

        edge_e = configEdges(
            scan,
            pre_edge_domain=pre_edge_domain,
            post_edge_domain=post_edge_domain,
            pre_edge_normalisation=edgeNormPre.EXPONENTIAL,
        )

        # Plot each normalisation step
        fig, ax = plt.subplots(1, 3, sharex=True, figsize=(15, 5))
        ax[0].plot(
            scan.x,
            scan.y[:, idx],
            label=f"{type(scan_mex2).__name__}\n{scan.y_labels[idx]}",
        )

        line = ax[1].plot(
            norm.x, norm.y[:, idx], label=f"{type(norm).__name__}\n{norm.y_labels[idx]}"
        )
        ax[1].plot(
            scan.x,
            scan.y[:, idx],
            "--",
            label=f"{type(scan_mex2).__name__}\n{scan.y_labels[idx]}",
        )  # c=l[0].get_color()

        line = ax[2].plot(
            edge_c.x,
            edge_c.y[:, idx],
            label=f"{type(edge_c).__name__}\nConstant\n{edge_c.y_labels[idx]}",
        )
        line = ax[2].plot(
            edge_l.x,
            edge_l.y[:, idx],
            label=f"{type(edge_l).__name__}\nLinear\n{edge_l.y_labels[idx]}",
        )
        line = ax[2].plot(
            edge_e.x,
            edge_e.y[:, idx],
            label=f"{type(edge_e).__name__}\nExponential\n{edge_e.y_labels[idx]}",
        )
        ax[1].plot(
            edge_e.x,
            configEdges.EXP_FN_OFFSET(
                edge_e.x, *edge_e.pre_edge_fit_params[3 * idx : 3 * (idx + 1)]
            ),
            "--",
            c=line[0].get_color(),
            label="Exponential Subtraction",
        )

        ax[2].set_xlabel("Energy (eV)")
        for a in ax:
            a.set_ylabel("Intensity (a.u.)")
            a.legend()

        ax[0].set_title("Original")
        ax[1].set_title("Normalised to I0")
        ax[2].set_title("Normalised to Edges")
        # ax[2].set_yscale("log")
        fig.tight_layout()
        plt.show()
