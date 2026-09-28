from typing import Literal, override

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import backend_tools, cbook, colormaps, colors
from matplotlib.patches import Rectangle
from matplotlib.widgets import (
    SpanSelector,
    ToolLineHandles,
    _call_with_reparented_event,
    _SelectorWidget,
)


class SpanSelectorN(SpanSelector):
    """
    A modified version of SpanSelector that allows for multiple selections.
    Visually select multiple min/max ranges on a single axis and call a singular
    or indexed function with those values.

    If 'interactive' is set to True, the widget will allow for the selection
    and movement of multiple handles in order. The selections can be removed by
    press/releasing on the same location in proximity to the span
    If 'interactive' is set to False, the widget will allow for N selections before
    clearing all selections.

    Notably `onselect` and `onmove_callback` can be singular or a list of callables,
    where the index of the callable will correspond to the index of the selection.
    If singular, the function will be called for all selections.

    To guarantee that the selector remains responsive, keep a reference to it.

    In order to turn off the SpanSelector, set ``span_selector.active`` to
    False. To turn it back on, set it to True.

    Press and release events triggered at the same coordinates outside the
    selection will clear the closest selector, except when
    ``ignore_event_outside=True``.

    `drag_from_anywhere` allows shifting a span by a press-drag event from
    anywhere within its bounds (the 'hand' cursor will be active).

    Escape key will clear all selections.

    Code based on matplotlib.widgets.SpanSelector.

    Parameters
    ----------
    N : int
        The number of selections that can be made.

    ax : `~matplotlib.axes.Axes`

    onselect : A singular callable or list of callables with signature
        ``func(min: float, max: float)``. The callback function is called
        after a release event, and the selection is created, changed or removed.
        If singular, the function will be called for all selections, otherwise
        the function at the corresponding index will be called.
        Using `callback_at_selection_complete=True` prevents the function(s)
        from being called until all N selections are made.

    direction : {"horizontal", "vertical"}
        The direction along which to draw the span selector.

    callback_at_selection_complete: bool, default: False
        If True, the onselect callback will be called only when the selection
        is completed, i.e., all N selections are made. If False, the callback
        will be called at each selection.

    minspan : float, default: 0
        If selection is less than or equal to *minspan*, the selection is
        removed (when already existing) or cancelled.

    useblit : bool, default: False
        If True, use the backend-dependent blitting features for faster
        canvas updates. See the tutorial :ref:`blitting` for details.

    props : dict, default: {'facecolor': 'red', 'alpha': 0.5}
        Dictionary of `.Patch` properties.
        Note `_default_colors_rect` from the colors argument will override the
        'facecolor' property, and 'alpha' if specified.

    onmove_callback : callable with signature ``func(min: float, max: float)``, optional
        Called on mouse move while the span is being selected.

    interactive : bool, default: False
        Whether to draw a set of handles that allow interaction with the
        widget after it is drawn.

    button : `.MouseButton` or list of `.MouseButton`, default: all buttons
        The mouse buttons which activate the span selector.

    handle_props : dict, default: None
        Properties of the handle lines at the edges of the span. Only used
        when *interactive* is True. See `.Line2D` for valid properties.
        Note `_default_colors_handles` from the colors argument will override
        the 'color' property in handle_props.

    grab_range : float, default: 10
        Distance in pixels within which the interactive tool handles can be activated.

    state_modifier_keys : dict, optional
        Keyboard modifiers which affect the widget's behavior.  Values
        amend the defaults, which are:

        - "clear": Clear the current shape, default: "escape".

    drag_from_anywhere : bool, default: False
        If `True`, the widget can be moved by clicking anywhere within its bounds.

    ignore_event_outside : bool, default: False
        If `True`, the event triggered outside the span selector will be ignored.

    snap_values : 1D array-like, optional
        Snap the selector edges to the given values.

    colors : :mpltype:`color` | list[:mpltype:`color`] | Colormap | None, optional
        A general mapping of colors to the spans and handles.
        By default uses mpl.colormaps.get("tab10").
        Can be a list of color names, a single color name, a list of RGBA tuples (0-1),
        a single RGBA tuple, or a matplotlib.colors.Colormap object.
        This option overrides the 'facecolor' property in *props* and the 'color'
        property in *handle_props*.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.widgets as mwidgets
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [10, 50, 100])
    >>> N=3
    >>> def onSelectorGenerator(i):
    ...     def onselect(vmin, vmax):
    ...         print(f"Span {i}:\t", vmin, vmax)
    ...     return onselect
    >>> fns = [onSelectorGenerator(i) for i in range(N)]
    >>> span = mwidgets.NSpanSelector(N, ax, fns, 'horizontal',
    ...                               useblit=True, interactive=True)
    >>> fig.show()

    See also: :doc:`/gallery/widgets/span_selectorN`
    """

    # overrides the __init__ method of the SpanSelector class
    # @overrides.overrides
    def __init__(
        self,
        N,
        ax,
        onselect,
        direction,
        *,
        callback_at_selection_complete=False,
        minspan=0,
        useblit=False,
        props={"alpha": 0.5},
        interactive=False,
        button=None,
        handle_props=None,
        grab_range=10,
        state_modifier_keys=None,
        drag_from_anywhere=False,
        ignore_event_outside=False,
        snap_values=None,
        onmove_callback=None,
        colors=colormaps.get("tab10"),
    ) -> None:
        # Define N
        self._N = N
        self._props = props  # rectangles
        self._handle_props = handle_props  # lines
        # Initialise tracking for active handle.
        self._active_selection_idx = None
        # Initialise callback behaviour for selection(s) completion.
        self._callback_at_selection_complete = callback_at_selection_complete
        # Store default colours
        self._default_colors_rect = colors
        self._default_colors_handles = None
        # Length check the onselect callback
        if hasattr(onselect, "__len__") and len(onselect) != N:
            raise ValueError(
                "`onselect` " + "must be a singular or a list of callables of length N."
            )
        if (
            onmove_callback is not None
            and hasattr(onmove_callback, "__len__")
            and len(onmove_callback) != N
        ):
            raise ValueError(
                "`onmove_callback` "
                + "must be a singular or a list of callables of length N."
            )
        super().__init__(
            ax=ax,
            onselect=onselect,
            direction=direction,
            minspan=minspan,
            useblit=useblit,
            props=props,
            interactive=interactive,
            grab_range=grab_range,
            drag_from_anywhere=drag_from_anywhere,
            ignore_event_outside=ignore_event_outside,
            button=button,
            onmove_callback=onmove_callback,
            handle_props=handle_props,
            state_modifier_keys=state_modifier_keys,
            snap_values=snap_values,
        )

        # Temporary fix to remove default red color and alpha from props,
        # which are applied in SpanSelector.__init__ and override the 'set'
        # of colors set here.
        if "facecolor" in self._props and self._props["facecolor"] == "red":
            del self._props["facecolor"]
        if "alpha" in self._props and self._props["alpha"] == 0.5:
            del self._props["alpha"]
        self._props.update(props)  # Re-apply user props.
        self.set_colors(colors, None)

    @property
    def N(self) -> int:
        """
        Property for the number of selections that can be made.

        Parameters
        ----------
        N : int
            The new number of selections to be made.
            If updating, `_default_colors_rect` and `_defualt_colors_handles`
            will be used to paint the new / truncated spans.

        Returns
        -------
        int
            The number of selections to be made.
        """
        return self._N

    @N.setter
    def N(self, N: int) -> None:
        if N != self._N:
            self._N = N
            # Update the axes again.
            self.new_axes(self.ax)

    ### --------------- Override attributes of _SelectorWidget ---------------:
    # Overrides to include self ._selection_artists instead of
    # self._selection_artist in the list of artists.
    @property
    def artists(self):
        handles_artists = getattr(self, "_handles_artists", ())
        return (*self._selection_artists,) + handles_artists

    # Overrides set_props to set properties for all selection artists.
    # Also updates the class dict for props.
    @override
    def set_props(self, index=None, **props) -> None:
        """
        Set properties for the selection artists.

        Parameters
        ----------
        index : int, optional
            If provided, only set the properties for the selection at the given index.
            Otherwise, set the properties for all selections, and update the global
            properties for all selections.
        **props
            Properties to set for the selection artists.
            See `.Patch` for valid properties.
        """
        artists = self._selection_artists
        if index is None:
            for artist in artists:
                props = cbook.normalize_kwargs(props, artist)
                artist.set(**props)
            # Additionally updates stores global selection props
            self._props.update(props)
        else:
            props = cbook.normalize_kwargs(props, artists[index])
            artists[index].set(**props)
            # Do not store local selection props.
        if self.useblit:
            self.update()

    def set_handle_props(self, index=None, **handle_props):
        """
        Set properties for the handle artists.

        Only applicable when interactive is True.

        Parameters
        ----------
        index : int, optional
            If provided, only set the properties for the handle at the given index.
            Otherwise, set the properties for all handles.
        **handle_props
            Properties to set for the handle artists.
            See `.Line2D` for valid properties.
        """
        artists = self._edge_handles.artists
        if artists is None or len(artists) == 0 or self._interactive is False:
            return

        if index is None:
            for artist in artists:
                handle_props = cbook.normalize_kwargs(handle_props, artist)
                artist.set(**handle_props)
            # Additionally updates stores global handle props
            self._handle_props.update(handle_props)
        else:
            handle_props = cbook.normalize_kwargs(handle_props, artists[2 * index])
            handle_props = cbook.normalize_kwargs(handle_props, artists[2 * index + 1])
            artists[2 * index].set(**handle_props)
            artists[2 * index + 1].set(**handle_props)
            # Do not store local handle props.
        if self.useblit:
            self.update()

    ### --------------- Overridden attributes of SpanSelector ---------------:
    # Overrides to define 2*self.N handles for selection in _edge_handles.
    @override
    def _setup_edge_handles(self, props=None) -> None:
        """
        Set up the edge handles for interactive selection.

        Parameters
        ----------
        props : dict, optional
            Properties for the handle lines. If None, uses the existing properties in
            self._handle_props. Note that the 'color' property will be overridden by
            _default_colors_handles if it is set.
        """

        # Define initial position using the axis bounds to keep the same bounds
        if self.direction == "horizontal":
            positions = self.ax.get_xbound()
        else:
            positions = self.ax.get_ybound()

        # Use N to define 2*N handles for selection in _edge_handles.
        dxy = (positions[1] - positions[0]) / (2 * self.N - 1)
        nPositions = (positions[0] + dxy * i for i in range(2 * self.N))

        # If existing handles, collect their positions to override
        handles = self._edge_handles
        if handles is not None:
            existing_positions = handles.positions
            nPositions = list(nPositions)
            for i in range(min(len(existing_positions), len(nPositions))):
                nPositions[i] = existing_positions[i]

        # Set handles.
        self._edge_handles = ToolLineHandles(
            self.ax,
            nPositions,
            direction=self.direction,
            line_props=props,
            useblit=self.useblit,
        )

        # Apply default colors to new handles
        if (
            self._default_colors_rect is not None
            and self._default_colors_handles is not None
        ):
            self.set_colors(self._default_colors_rect, self._default_colors_handles)

    # Overrides to define self.N rectangles for selection in rect_artists
    # and self._selection_artists variables. Previously was rect_artist and
    # self._selection_artist.
    @override
    def new_axes(self, ax, *, _props=None, _init=False) -> None:
        """
        Set SpanSelectorN to operate onto a new Axes.
        """
        # Also implements an axis update, where the number of selections can be changed.

        # Reset selection on new axes.
        self._selection_completed = False
        if (
            self.ax is ax
            and getattr(self, "_selection_artists", None) is not None
            # Legacy checks from SpanSelector:
            or _init
            or self.canvas is not ax.get_figure(root=True).canvas
        ):
            # Updating axis
            art_len = len(self._selection_artists)
            if art_len == self.N:
                return
            else:
                # Use existing artists if the Axes is the same.
                if art_len >= self.N:
                    self._selection_artists = self._selection_artists[: self.N]
                else:
                    # Create new artists if the number of selections has increased.
                    # Hide the artists at creation.
                    v1, v2 = self.ax.get_xbound()
                    if art_len > 0:
                        v1 = (
                            self._selection_artists[-1].get_x()
                            + self._selection_artists[-1].get_width()
                        )
                    new_artists = [
                        Rectangle(
                            xy=(
                                (
                                    (
                                        v1 + i
                                        if v1 + i < v2
                                        else v1 + ((v1 + i) % (v2 - v1))
                                    ),
                                    0,
                                )
                                if self.direction == "horizontal"
                                else (
                                    0,
                                    (
                                        v1 + i
                                        if v1 + i < v2
                                        else v1 + ((v1 + i) % (v2 - v1))
                                    ),
                                )
                            ),
                            width=0 if self.direction == "horizontal" else 1,
                            height=1 if self.direction == "horizontal" else 0,
                            transform=(
                                self.ax.get_xaxis_transform()
                                if self.direction == "horizontal"
                                else self.ax.get_yaxis_transform()
                            ),
                            visible=False,
                        )
                        for i in range(self.N - art_len)
                    ]
                    for i, rect in enumerate(new_artists):
                        # Add to axes
                        self.ax.add_patch(rect)
                        # Apply props
                        if self._props is not None:
                            rect.update(self._props)
                    # Color the axes
                    if self._default_colors_rect is not None:
                        self.set_colors(
                            self._default_colors_rect, self._default_colors_handles
                        )
                    # Append to selection artists
                    self._selection_artists += new_artists
                # Re-associate artists with axes.
                self._setup_edge_handles()
            # Disconnect and re-connect events to update canvas.
            self.disconnect_events()
            self.connect_default_events()
        else:
            self.ax = ax
            # assert isinstance(self.ax, Axes) # allow editor to recognise type.
            if self.canvas is not None:
                self.disconnect_events()
            if self.canvas is not ax.figure.canvas:  # Why is this necessary?
                self.canvas = ax.figure.canvas
            self.connect_default_events()

            # Direction of variables.
            if self.direction == "horizontal":
                trans = ax.get_xaxis_transform()
                w, h = 0, 1
            else:
                trans = ax.get_yaxis_transform()
                w, h = 1, 0

            # Generate rectangles for selection.
            # use double the number of selections to allow spaces between

            dw = w / (2 * self.N - 1)
            dh = h / (2 * self.N - 1)
            rect_artists = [
                Rectangle(
                    xy=(
                        (2 * i * dw, 0)
                        if self.direction == "horizontal"
                        else (0, 2 * i * dh)
                    ),
                    width=w,
                    height=h,
                    transform=trans,
                    visible=False,
                )
                for i in range(self.N)
            ]
            for i, rect_artist in enumerate(rect_artists):
                if _props is not None:
                    rect_artist.update(_props)
                elif self._selection_artists is not None and len(
                    self._selection_artists
                ) == len(rect_artists):
                    rect_artist.update_from(self._selection_artists[i])
                self.ax.add_patch(rect_artist)
            self._selection_artists = rect_artists

    # Overrides _press to handle multiple selections.
    # Also adjusts extents for only the active handle.
    # Behaviour modified to handle removing and adding selections, and dragging.
    # Added _active_handle_vis to track if the handle was previously visible or not.
    # Added _active_selection_idx to track the active selection index.
    @override
    def _press(self, event) -> Literal[False]:
        """Button press event handler."""
        xdata, ydata = self._get_data(event)
        v = xdata if self.direction == "horizontal" else ydata
        self._set_cursor(
            backend_tools.cursors.RESIZE_HORIZONTAL
            if self.direction == "horizontal"
            else backend_tools.cursors.RESIZE_VERTICAL
        )

        # Tracking for non-visible selection artists.
        # Used in self._release if the selection is less than minspan;
        # Removes nearest visible instead of revealing invisible.
        self._active_handle_vis: bool = True
        """True (False) if the active handle was (in-)visible at press time."""

        if self._edge_handles is not None:
            # Check if  hovering an existing visible handle.
            index, e_dist = self._edge_handles.closest(event.x, event.y)
            hover = (
                # Check if within grab range of an edge handle.
                e_dist <= self.grab_range
                and self._edge_handles.artists[index].get_visible()
            ) or self._contains(event)

            # # Check if within any existing selection.
            # within, within_idx = self._within_handle(event)
            # if within:
            #     hover = True

        else:
            # Handle unset edge handles case.
            index, e_dist = 0, 0
            hover = False

        # If any artists rect selection artists are not visible,
        # add one at the cursor location
        sel_artists_vis = np.array(
            [artist.get_visible() for artist in self._selection_artists], bool
        )
        if (self._visible is False or np.any(~sel_artists_vis)) and not hover:
            i = np.argmin(sel_artists_vis)
            ex = self.extents
            ex[i] = v, v
            self.extents = ex

            self._active_handle_vis = False
            self.set_visible(True, i)
            sel_artists_vis[i] = True  # Update visibility array

            # Setting active handle when no handles are defined
            # (ie. self._interactive = False) is necessary.
            self._active_selection_idx = i

        # Set the active handle based on the location of the mouse event.
        if any(sel_artists_vis) and self._interactive:
            self._set_active_handle(event)
        elif any(sel_artists_vis) and self._active_selection_idx is not None:
            # When not interactive, but active handle index has been defined.
            self._active_handle = None
        else:
            self._active_handle = None
            self._active_selection_idx = None

        # If no handle is active, then we are done.
        if self._active_handle is None or not self._interactive:
            # Clear previous rectangle before drawing new rectangle.
            self.update()
        return False

    # Overrides to track which handle index is active.
    def _set_active_handle(self, event):
        """Set active handle based on the location of the mouse event."""
        # Note: event.xdata/ydata in data coordinates, event.x/y in pixels
        e_idx, e_dist = self._edge_handles.closest(event.x, event.y)
        h_idx = int(np.floor(e_idx / 2))  # Handle index
        edge_order_idx = e_idx % 2  # 0 for min, 1 for max

        # Save selection index. 2 handles per selection, so divide by 2.
        self._active_selection_idx = h_idx

        ## Prioritise within proximity to edge handle, then centre handle, then outside.
        if e_dist < self.grab_range:
            # Closest to an edge handle
            self._active_handle = self._edge_order[edge_order_idx]
        # Note: self._contains changes self._active_handle_idx if not True
        # for self._active_handle_idx but True for a different index.
        elif self.drag_from_anywhere and self._contains(event):
            # Check if we've clicked inside any region.
            # Update the extents, in case the active handle index has changed.
            h_idx = self._active_selection_idx
            self._active_handle = "C"
        elif "move" in self._state:
            self._active_handle = "C"
        elif not self.ignore_event_outside:
            # If outside the region, instead use closest edge determine which selection.
            self._active_handle = self._edge_order[edge_order_idx]
            # Get the extents for the active handle.
            xdata, ydata = self._get_data(event)
            v = xdata if self.direction == "horizontal" else ydata
            ex = self.extents
            ex[h_idx] = v, v
            self.extents = ex
        else:
            self._active_handle = None
            self._active_selection_idx = None

        # Save coordinates of rectangle at the start of handle movement.
        self._extents_on_press = self.extents[h_idx]

    # Overrides to check if event is within any selection artist,
    # and changes the active index.
    def _contains(self, event):
        """
        Checks if event is within a visible selection artist.

        Return True if event is within the selected handle index,
        or within another selection artist, updating the active index.
        """
        i = self._active_selection_idx
        # if i is None:
        # return False
        # else:
        if (
            i is not None
            and self._selection_artists[i].get_visible()
            and self._selection_artists[i].contains(event, radius=0)[0]
        ):
            return True
        # If the event is within another span,
        # then the active handle index is updated.
        # else:
        # Loop through other selection artists to check if event is within any.
        for j, selection_artist in enumerate(self._selection_artists):
            if (
                j != i
                and selection_artist.get_visible()
                and selection_artist.contains(event, radius=0)[0]
            ):
                # Update active handle index to the
                # selection artist that contains the event.
                self._active_selection_idx = j
                return True

    # Overrides to handle multiple selections in movement.
    def _onmove(self, event):
        """Motion notify event handler."""

        xdata, ydata = self._get_data(event)
        if self.direction == "horizontal":
            v = xdata
            vpress = self._eventpress.xdata
        else:
            v = ydata
            vpress = self._eventpress.ydata

        # move existing span
        # When "dragging from anywhere", `self._active_handle` is set to 'C'
        # (match notation used in the RectangleSelector)
        if self._active_handle == "C" and self._extents_on_press is not None:
            vmin, vmax = self._extents_on_press
            dv = v - vpress
            vmin += dv
            vmax += dv

        # resize an existing shape
        elif (
            self._active_handle
            and self._active_handle != "C"
            and self._extents_on_press is not None
        ):
            vmin, vmax = self._extents_on_press
            if self._active_handle == "min":
                vmin = v
            else:
                vmax = v
        # new shape
        else:
            # Don't create a new span if there is already one when
            # ignore_event_outside=True
            if self.ignore_event_outside and self._selection_completed:
                return
            vmin, vmax = vpress, v

        # Swap vmin, vmax if necessary
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        if self._active_selection_idx is not None:
            ex = self.extents
            ex[self._active_selection_idx] = vmin, vmax
            self.extents = ex

        if self.onmove_callback is not None:
            # If a list corresponding to spans, call particular span.
            if isinstance(self.onmove_callback, list):
                if self._active_selection_idx is not None:
                    self.onmove_callback[self._active_selection_idx](vmin, vmax)
                else:
                    # No index selected, don't call.
                    pass
            else:
                # Else singular function for all spans.
                self.onmove_callback(vmin, vmax)
        return False

    # Overrides to check if hovering over any visible edge handle.
    # Additionally enables functionality with incomplete selection.
    @override
    @_call_with_reparented_event
    def _hover(self, event):
        """Update the canvas cursor if it's over a handle."""
        if self.ignore(event):
            return

        if self._active_handle is not None:
            # Do nothing if button is pressed and a handle is active, which may
            # occur with drag_from_anywhere=True.
            return

        index, e_dist = self._edge_handles.closest(event.x, event.y)
        # Within proximity?
        if (
            e_dist <= self.grab_range
            and self._edge_handles.artists[index].get_visible()
        ):
            self._set_cursor(
                # Within grab range of a visible handle
                backend_tools.cursors.RESIZE_HORIZONTAL
                if self.direction == "horizontal"
                else backend_tools.cursors.RESIZE_VERTICAL
            )
        else:
            # Check if within any span
            contains = self._contains(event)
            self._set_cursor(
                backend_tools.cursors.HAND
                if contains and self.drag_from_anywhere
                else backend_tools.cursors.POINTER
            )

    # Overrides to handle multiple selections, by adding an index parameter.
    # Same as SpanSelector.set_visible for all artists if no index is provided.
    def set_visible(self, visible: bool, index: int | None = None) -> None:
        """
        Overrides functionality of default _SelectorWidget.set_visible method.
        Uses additional artists and sets visibility for all selection artists
        if no index is provided.

        Parameters
        ----------
        visible : bool
            The visibility of the selection artists.
        index : int, optional
            The index of the selection artist to set visibility for. If None, then
            all selection artists will be set to the same visibility.
        """
        if index is not None:
            if len(self._selection_artists) <= index:
                raise IndexError(
                    f"Index {index} is out of bounds for selection artists of length "
                    + f"{len(self._selection_artists)}."
                )
            elif len(self._edge_handles.artists) <= index * 2 + 1:
                raise IndexError(
                    f"Index {index} is out of bounds for edge handle artists of length "
                    + f"{len(self._edge_handles.artists)}."
                )

            self._selection_artists[index].set_visible(visible)
            if self._edge_handles is not None and self._interactive:
                self._edge_handles._artists[index * 2].set_visible(visible)
                self._edge_handles._artists[index * 2 + 1].set_visible(visible)
            if visible is True:
                self._visible = True
                return
            else:
                if np.all([artist.get_visible() is False for artist in self.artists]):
                    self._visible = False
        else:
            self._visible = visible
            for artist in self.artists:
                artist.set_visible(visible)

    # Overrides release method to call onselect for various selections.
    @override
    def _release(self, event):
        """Button release event handler."""
        self._set_cursor(backend_tools.cursors.POINTER)

        ### If already completed, return and don't trigger onselect callback.
        if (
            self._active_handle is None
            and self._selection_completed
            and self.ignore_event_outside
        ):
            return

        ### Get the extents
        ext = self.extents
        spans = [vmax - vmin for vmin, vmax in ext]

        ### Call function(s) for selections.
        # If not interactive (ie, spans disappear after full selection)
        # then hide once the last selection is made.
        if not self._interactive:
            # Check if the active handle belongs to the last required selection
            # i.e. last two handles.
            if (
                self._active_selection_idx is not None
                and self._active_selection_idx >= self.N - 1
            ):
                for artist in self._selection_artists:
                    artist.set_visible(False)

        # Hide current span (or nearest visible span) in case the span is less than
        # minspan. Perform again later if changing handle. If active handle wasn't
        # previously visible, hide it and instead hide another nearest visible
        # selection.
        idx = self._active_selection_idx
        if idx is not None:
            if spans[idx] <= self.minspan:
                self.set_visible(False, index=idx)

        # Find visible selection artists, excluding the active handle.
        visible_rects = [
            select_artist
            for i, select_artist in enumerate(self._selection_artists)
            if select_artist.get_visible() and i != idx
        ]
        # Check if active selection wasn't and won't be visible
        if (
            not self._active_handle_vis
            and idx is not None
            and spans[idx] <= self.minspan
            and len(visible_rects) > 0
        ):
            # Hide the nearest visible selection if any visible.
            # Use release x,y to find closest visible.
            xdata, ydata = self._get_data(event)
            if xdata is not None and ydata is not None:
                vis_idx = np.argmin(
                    [
                        # Minimum of X or X + Width for horizontal
                        (
                            min(
                                abs(select_artist.get_x() - xdata),
                                abs(
                                    select_artist.get_x()
                                    + select_artist.get_width()
                                    - xdata
                                ),
                            )
                            if self.direction == "horizontal"
                            # Minimum of Y or Y + Height for vertical
                            else min(
                                abs(select_artist.get_y() - xdata),
                                abs(
                                    select_artist.get_y()
                                    + select_artist.get_height()
                                    - ydata
                                ),
                            )
                        )
                        for select_artist in visible_rects
                    ]
                )
                # Find the index of the visible selection in the selection artists.
                closest_idx = self._selection_artists.index(visible_rects[vis_idx])
                # Set the nearest visible selection to invisible.
                self.set_visible(False, index=closest_idx)
                # Update the extents to reflect the zero-width, hidden selection.
                ext[closest_idx] = (
                    (xdata, xdata) if self.direction == "horizontal" else (ydata, ydata)
                )
                spans[closest_idx] = 0
                # Change the active handle index to freshly hidden selection.
                self._active_selection_idx = closest_idx
                self._active_handle_vis = True

        # Perform the selection call.
        if (
            idx is not None and self.onselect
        ):  # Requires active handle, and non-None onselect.
            idx = self._active_selection_idx  # active handle may have changed.
            vmin, vmax = ext[idx]
            # Check if the current span is being hidden but was visible.
            if (
                spans[idx] <= self.minspan
                and self._selection_artists[idx] in visible_rects
            ):
                # Only callback when selection complete?
                if not self._callback_at_selection_complete:
                    if isinstance(self.onselect, list):
                        if callable(self.onselect[idx]):
                            self.onselect[idx](vmin, vmax)
                        else:
                            raise ValueError(
                                f"onselect `{self.onselect[idx]}` must be a callable."
                            )
                    else:
                        if callable(self.onselect):
                            self.onselect(vmin, vmax)
                        else:
                            raise ValueError(
                                f"onselect `{self.onselect}` must be a callable."
                            )
            elif spans[idx] > self.minspan:
                if self._callback_at_selection_complete:
                    rect_artists = self._selection_artists
                    if all(artist.get_visible() for artist in rect_artists):
                        # Perform callbacks when all selections are visible.
                        for i, (wmin, wmax) in enumerate(ext):
                            if isinstance(self.onselect, list):
                                # If onselect is a list, call the indexed function.
                                if callable(self.onselect[idx]):
                                    self.onselect[i](wmin, wmax)
                                else:
                                    raise ValueError(
                                        f"onselect `{self.onselect[i]}` "
                                        + "must be a callable."
                                    )
                            else:
                                if callable(self.onselect):
                                    self.onselect(wmin, wmax)
                                else:
                                    raise ValueError(
                                        f"onselect `{self.onselect}` "
                                        + "must be a callable."
                                    )
                else:
                    # Singular calls for each selection.
                    if isinstance(self.onselect, list):
                        # If onselect is a list, call the indexed function.
                        if callable(self.onselect[idx]):
                            self.onselect[idx](vmin, vmax)
                        else:
                            raise ValueError(
                                f"onselect `{self.onselect[idx]}` must be a callable."
                            )
                    else:
                        if callable(self.onselect):
                            self.onselect(vmin, vmax)
                        else:
                            raise ValueError(
                                f"onselect `{self.onselect}` must be a callable."
                            )
        else:
            # Do not call for incomplete selections,
            # or hidden selections remaining hidden.
            pass

        # If all selections are completed, then set _selection_completed to True.
        self._selection_completed = all(
            span.get_visible() for span in self._selection_artists
        )

        self.update()

        # Reset active handle
        self._active_handle = None
        self._active_selection_idx = None
        self._active_handle_vis = None
        self._extents_on_press = None

        return False

    @override
    def connect_default_events(self):
        # docstring inherited
        _SelectorWidget.connect_default_events(self)
        if getattr(self, "_interactive", False):
            self.connect_event("motion_notify_event", self._hover)

    @property
    def extents(self) -> list[tuple[float, float]]:
        """
        The extents of all span selections.

        Parameters
        ----------
        extents : list[tuple[float, float]]
            The values, in data coordinates, for the start and end points
            of all span selections. If there is no selection then the
            start and end values will be the same.

        Returns
        -------
        list[tuple[float, float]]
            The values, in data coordinates, for the start and end points
            of all span selections. If there is no selection then the
            start and end values will be the same.
        """
        # Overrides to use self._selection_artists instead of self._selection_artist.
        vals = []
        for selection_artist in self._selection_artists:
            if self.direction == "horizontal":
                vmin = selection_artist.get_x()
                vmax = vmin + selection_artist.get_width()
            else:
                vmin = selection_artist.get_y()
                vmax = vmin + selection_artist.get_height()
            vals.append((vmin, vmax))
        return vals

    @extents.setter
    def extents(self, extents: list[tuple[float, float]]):
        if len(extents) > self.N:
            raise ValueError("Number of extents must be equal to or less than N.")
        # Update displayed shape
        if self.snap_values is not None:
            for i, extent in enumerate(extents):
                extents[i] = tuple(self._snap(extent, self.snap_values))
        # Draw the physical shapes
        self._draw_shapes(extents)
        if self._interactive:
            # Update displayed handles
            extent_data = np.array(extents).flatten()
            self._edge_handles.set_data(extent_data)
        self.update()

    def _draw_shapes(
        self,
        extents: list[tuple[float, float]],
    ) -> None:
        """
        Draws the selection shapes on the axes.

        An alternative method to SpanSelector._draw_shape.

        Parameters
        ----------
        extents : list[tuple[float, float]]
            The extents for each selection shape to be drawn.
        """
        for i, extent in enumerate(extents):
            vmin, vmax = extent
            # Reorder if necessary
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            if self.direction == "horizontal":
                self._selection_artists[i].set_x(vmin)
                self._selection_artists[i].set_width(vmax - vmin)
            else:
                self._selection_artists[i].set_y(vmin)
                self._selection_artists[i].set_height(vmax - vmin)

    def set_colors(
        self,
        rect_colors,  # list["ColorType"] | "ColorType" | colors.Colormap,
        handle_colors=None,  #: list["ColorType"] | "ColorType"
        # | colors.Colormap | None = None,
        update_defaults: bool = True,
        dim: float = 0.5,
    ) -> None:
        """
        Set the colors of the rectangle spans and their edge handles.

        Also updates the default colors each time handles are created.

        Parameters
        ----------
        colors : list[:mpltype:`color`] | :mpltype:`color` | Colormap
            A general mapping of colors to the spans.
            If a list is provided, it should match the length of N.
            If a single color is provided, all spans will be set to that color.
            If a Colormap is provided, LinearSegmentedColormap will be used to
            generate N colors from the colormap, while ListedColormap will be truncated
            or repeated to match N. Values can be a string (i.e. hex), a tuple of RGBA
            values (0-1), or a Colormap.

        handle_colors : list[:mpltype:`color`] | :mpltype:`color` | Colormap, optional
            A general mapping of colors to the edge handles. By default None.
            Should match the dimensions of `colors` parameter if a list.
            If None, the edge handles will be multiplied by `dim` to modify the RGB
            values of the rectangle colors. Note, handle alpha values are
            disregarded and inherited from the rectangle colors.

        update_defaults : bool, default: True
            Whether to update the default colors used when creating new axes.

        dim : float, default: 0.5
            The factor by which to dim the RGB values of the rectangle colors
            to create the edge handle colors, if `handle_colors` is None.
        """
        rect_clist: list = []
        """The list of colors for the rectangles."""
        handle_clist: list = []
        """The list of modified colors for the edge handles."""

        ## Gather all colors for the rectangles:
        # Colormap
        if isinstance(rect_colors, colors.Colormap):
            # Choose evently spaced N colors from a LinearSegmentedColormap,
            # or first N colors from a ListedColormap.
            if isinstance(rect_colors, colors.ListedColormap):
                # Truncate or repeat ListedColormap to match N
                base_colors = rect_colors.colors

                if hasattr(base_colors, "__len__"):
                    n_base = len(base_colors)
                    for i in range(self.N):
                        c = base_colors[i % n_base]
                        # RGB/RGBA tuple
                        if (
                            isinstance(c, tuple)
                            and (len(c) == 3 or len(c) == 4)
                            and all(isinstance(v, (int, float)) for v in c)
                        ):
                            rect_clist.append(c)  # type: ignore
                        elif isinstance(c, str):
                            rect_clist.append(c)
                        else:
                            raise TypeError(
                                "ListedColormap contains an invalid "
                                f"color format ({c})."
                            )
                else:
                    # No length, assume gray scale value
                    rect_clist = [base_colors for _ in range(self.N)]  # type: ignore
            elif isinstance(rect_colors, colors.LinearSegmentedColormap):
                # Use LinearSegmentedColormap to get N colours.
                n_list = np.linspace(0, 1, self.N)
                rect_clist = rect_colors(n_list).tolist()
            # TODO: Do we require support for var-cmaps?
            # elif isinstance(rect_colors,
            # (colors.BivarColormap, colors.MultivarColormap)):
            else:
                raise TypeError(
                    "rect_colors Colormap must be "
                    "ListedColormap or LinearSegmentedColormap."
                )
        # LIST
        elif isinstance(rect_colors, list):
            rect_clist = rect_colors.copy()
        else:
            raise TypeError("rect_colors must be a list, tuple, string, or Colormap.")

        ## Gather all colors for the edge handles:
        # Colormap
        if isinstance(handle_colors, colors.Colormap):
            # Choose evently spaced N colors from a LinearSegmentedColormap,
            # or first N colors from a ListedColormap.
            if isinstance(handle_colors, colors.ListedColormap):
                # Truncate or repeat ListedColormap to match N
                base_colors = handle_colors.colors
                # Make sure base_colors is a list
                if not isinstance(base_colors, (list, np.ndarray)):
                    base_colors = [base_colors]
                n_base = len(base_colors)
                for i in range(self.N):
                    c = base_colors[i % n_base]
                    # RGB/RGBA tuple
                    if (
                        isinstance(c, tuple)
                        and (len(c) == 3 or len(c) == 4)
                        and all(isinstance(v, (int, float)) for v in c)
                    ):
                        handle_clist.append(c)  # type: ignore
                    elif isinstance(c, str):
                        handle_clist.append(c)
                    else:
                        raise TypeError(
                            f"ListedColormap contains an invalid color format ({c})."
                        )
            elif isinstance(handle_colors, colors.LinearSegmentedColormap):
                # Use LinearSegmentedColormap to get N colours.
                n_list = np.linspace(0, 1, self.N)
                handle_clist = handle_colors(n_list).tolist()
            else:
                raise TypeError(
                    "handle_colors Colormap must be ListedColormap "
                    "or LinearSegmentedColormap."
                )
        # LIST
        elif isinstance(handle_colors, list):
            handle_clist = handle_colors.copy()
        # Default None, Generate from dimming rect_colors
        elif handle_colors is None:
            for color in rect_clist:
                # Hex / string colors
                if isinstance(color, str):
                    # Convert to RGB tuple, then halve RGB values for edge handles.
                    rgb = colors.to_rgb(color)
                    handle_clist.append((dim * rgb[0], dim * rgb[1], dim * rgb[2]))
                # RGB / RGBA tuple colors
                elif isinstance(color, (tuple, list)) and all(
                    isinstance(c, (int, float)) for c in color
                ):
                    # RGBA
                    if len(color) == 4:
                        handle_clist.append(
                            (dim * color[0], dim * color[1], dim * color[2], color[3])
                        )
                    # RGB
                    elif len(color) == 3:
                        handle_clist.append(
                            (dim * color[0], dim * color[1], dim * color[2])
                        )
                    else:
                        raise TypeError(
                            "rect_colors contains an invalid color format "
                            f"of float/ints with length {len(color)}."
                        )
                # Tuple of (color, alpha)
                elif (
                    isinstance(color, (tuple, list))
                    and isinstance(color[0], (str, tuple))
                    and isinstance(color[1], (int, float))
                    and len(color) == 2
                ):
                    # (str, alpha)
                    if isinstance(color[0], str):
                        rgb = colors.to_rgb(color[0])
                        handle_clist.append(
                            (dim * rgb[0], dim * rgb[1], dim * rgb[2], color[1])
                        )
                    # (RGB/RGBA tuple, alpha)
                    elif (
                        isinstance(color[0], (tuple, list))
                        and all(isinstance(c, (int, float)) for c in color[0])
                        and (len(color[0]) == 3 or len(color[0]) == 4)
                    ):
                        # Override alpha if RGBA tuple
                        handle_clist.append(
                            (
                                dim * color[0][0],
                                dim * color[0][1],
                                dim * color[0][2],
                                color[1],
                            )
                        )
                    else:
                        raise TypeError(
                            f"rect_colors contains an invalid color "
                            f"format ({color}) in a (color, alpha) tuple."
                        )
                else:
                    raise TypeError(
                        f"rect_colors contains an invalid color format ({color})."
                    )

            def clamp(rgb: tuple[float, ...]) -> tuple[float, ...]:
                """Clamp all RGB/RGBA values to be within 0-1 range."""
                clamped = []
                for v in rgb:
                    if v < 0:
                        clamped.append(0)
                    elif v > 1:
                        clamped.append(1)
                    else:
                        clamped.append(v)
                return tuple(clamped)

            # Ensure all handle clist are within valid range.
            for i, color in enumerate(handle_clist):
                # RGBA/RGB/(RGBA/RGB, alpha) tuple
                if isinstance(color, tuple):
                    if (
                        len(color) == 2
                        and isinstance(color[0], tuple)
                        and len(color[0]) >= 3
                    ):
                        # (RGB|RGBA, alpha) tuple
                        color = clamp(tuple([*color[0:3], color[1]]))
                        handle_clist[i] = color
                    elif (len(color) == 3 or len(color) == 4) and all(
                        isinstance(c, (int, float)) for c in color
                    ):
                        # RGB|RGBA tuple
                        color = clamp(color)
                        handle_clist[i] = color
                    else:
                        raise TypeError(
                            "handle_colors contains an invalid color"
                            "format of float/ints with "
                            f"length {len(color)}."
                        )
                # No need to handle string colors.
        else:
            raise TypeError("handle_colors must be a list, tuple, string, or Colormap.")

        ## Apply colours to the rectangles and edge handles.
        for i, selection_artist in enumerate(self._selection_artists):
            color = rect_clist[i % len(rect_clist)]
            selection_artist.set_facecolor(color)
            # Does the color have an alpha value?
            if isinstance(color, tuple) and len(color) == 4:
                selection_artist.set_alpha(color[3])
            elif isinstance(color, tuple) and len(color) == 2:
                selection_artist.set_alpha(color[1])
            # Set edge handle colors
            if self._interactive:
                hcolor = handle_clist[i % len(handle_clist)]
                self._edge_handles._artists[i * 2].set_color(color=hcolor)
                self._edge_handles._artists[i * 2 + 1].set_color(color=hcolor)

                # Handle alpha is controlled by the rect alpha.
                if isinstance(hcolor, tuple):
                    if len(hcolor) == 2:
                        self._edge_handles._artists[i * 2].set_alpha(hcolor[1])
                        self._edge_handles._artists[i * 2 + 1].set_alpha(hcolor[1])
                    elif len(hcolor) == 4:
                        self._edge_handles._artists[i * 2].set_alpha(hcolor[3])
                        self._edge_handles._artists[i * 2 + 1].set_alpha(hcolor[3])

        # Update the default colors.
        if update_defaults:
            self._default_colors_rect = rect_clist
            self._default_colors_handles = handle_clist

    @property
    def colors_rect(self):
        """
        Property to get/set the colours of the span rectangles.

        Updates the `_default_colors_rect` values.

        Parameters
        ----------
        colors : list[:mpltype:`color`] | :mpltype:`color`
            List of colours to set for each span rectangle.

        Returns
        -------
        list[:mpltype:`color`] | :mpltype:`color`]
            List of colours for each rectangle, or single colour if N=1.
        """
        result = [artist.get_facecolor() for artist in self._selection_artists]
        if self.N == 1:
            return result[0]
        return result

    @colors_rect.setter
    def colors_rect(self, cols):
        if isinstance(cols, list):
            for i, color in enumerate(cols):
                if not colors.is_color_like(color):
                    raise ValueError(f"color '{color}' is not a valid color")
                # Set alpha if provided
                # if isinstance(color, )
                if len(color) > 3:
                    self._selection_artists[i].set_alpha(color[3])
                self._selection_artists[i].set_facecolor(color[0:3])
        elif isinstance(cols, str):
            for artist in self._selection_artists:
                if len(color) > 3:
                    artist.set_alpha(color[3])
                artist.set_facecolor(color[0:3])
        else:
            raise ValueError(f"colors '{cols}' must be a list or a string")
        self._default_colors_rect = cols
        self.update()

    @property
    def colors_handles(self) -> list[tuple[str, str]]:
        """
        Property to get the colours of the span edge handles.

        Also updates the `_default_handle_colors`.

        Parameters
        ----------
        colors : list[tuple[str,str]]
            List of colours to set for each span edge handle.

        Returns
        -------
        list[tuple[str,str]]
            List of colours for each span edge handle
        """
        colors = [artist.get_color() for artist in self._edge_handles.artists]
        colors = [(colors[i], colors[i + 1]) for i in range(0, len(colors), 2)]
        return colors

    @colors_handles.setter
    def colors_handles(self, colors: list[tuple[str, str]] | tuple[str, str] | str):
        if isinstance(colors, list):
            for i, color in enumerate(colors):
                self._edge_handles.artists[i * 2].set_color(color[0])
                self._edge_handles.artists[i * 2 + 1].set_color(color[1])
        if isinstance(colors, tuple):
            for i, artist in enumerate(self._edge_handles.artists):
                if i % 2 == 0:
                    artist.set_color(colors[0])
                else:
                    artist.set_color(colors[1])
        elif isinstance(colors, str):
            for artist in self._edge_handles.artists:
                artist.set_color(colors)
        else:
            raise ValueError(
                f"colors {colors} must be a list of tuples, a tuple or a string"
            )
        self._default_colors_handles = colors
        self.update()


if __name__ == "__main__":
    mpl.use("QtAgg")
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)

    span = 5

    def onselect1(xmin, xmax):
        print("1st Span:\t", xmin, xmax)

    def onselect2(xmin, xmax):
        print("2nd Span:\t", xmin, xmax)

    def onselect3(xmin, xmax):
        print("3rd Span:\t", xmin, xmax)
        thecols = span.colors_rect
        newGreen = np.sqrt(thecols[-1])
        newGreen[0:3] = thecols[-1][0:3]  # keep colour, change alpha
        thecols[-1] = tuple(newGreen)
        span.colors_rect = thecols

    span = SpanSelectorN(
        N=3,
        ax=ax,
        onselect=[onselect1, onselect2, onselect3],
        direction="horizontal",
        drag_from_anywhere=True,
        interactive=True,
        useblit=True,
    )

    # span = SpanSelector(
    #     ax=ax,
    #     onselect=onselect1,
    #     direction="horizontal",
    #     drag_from_anywhere=True,
    #     interactive=True,
    #     useblit=True,
    # )

    plt.show()
