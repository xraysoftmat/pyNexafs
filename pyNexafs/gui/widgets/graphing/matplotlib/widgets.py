import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import SpanSelector, ToolLineHandles
from matplotlib.patches import Rectangle
from matplotlib.colors import Colormap
import numpy as np
import numpy.typing as npt
from typing import Any, Collection, Literal, Callable
import overrides


class NSpanSelector(SpanSelector):
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
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

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

    colors : list[str] | str | list[tuple[float, float, float, float]] | tuple[float, float, float, float] | Colormap | None, optional
        A general mapping of colors to the spans. By default uses mpl.colormaps.get("tab10").


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
    >>> span = mwidgets.NSpanSelector(N, ax, fns, 'horizontal', useblit=True, interactive=True)
    >>> fig.show()

    See also: :doc:`/gallery/widgets/span_selector`
    """

    # overrides the __init__ method of the SpanSelector class
    # @overrides.overrides
    def __init__(
        self,
        N: int,
        ax: Axes,
        onselect: list[Callable[[float, float], Any]] | Callable[[float, float], Any],
        direction: Literal["horizontal"] | Literal["vertical"],
        callback_at_selection_complete: bool = False,
        minspan: float = 0,
        useblit: bool = False,
        props: dict[str, Any] | None = {"facecolor": "red", "alpha": 0.5},
        interactive: bool = False,
        grab_range: float = 10,
        drag_from_anywhere: bool = False,
        ignore_event_outside: bool = False,
        button: MouseButton | Collection[MouseButton] | None = None,
        onmove_callback: (
            list[Callable[[float, float], Any]] | Callable[[float, float], Any] | None
        ) = None,
        handle_props: dict[str, Any] | None = None,
        state_modifier_keys: dict[str, str] | None = None,
        snap_values: npt.ArrayLike = None,
        colors: (
            list[str]
            | str
            | list[tuple[float, float, float, float]]
            | tuple[float, float, float, float]
            | Colormap
        ) = mpl.colormaps.get("tab10"),
    ) -> None:
        # Define N
        self._N = N
        self._props = props  # rectangles
        self._handle_props = handle_props  # lines
        # Initialise tracking for active handle.
        self._active_selection_idx = None
        # Initialise callback behaviour for selection(s) completion.
        self._callback_at_selection_complete = callback_at_selection_complete
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
        self.color_spans(colors)

    @property
    def N(self) -> int:
        """Returns the number of selections that can be made."""
        return self._N

    @N.setter
    def N(self, N: int) -> None:
        """Sets the number of selections that can be made."""
        if N != self._N:
            # Update the number of edge handles based on the change from old self._N
            if self._edge_handles is not None:
                # Get existing positions
                positions = self._edge_handles.positions
                if N > self._N:
                    # Add new positions to the end, by default invisible.
                    ax_v1 = positions[-1]
                    nPositions = (
                        *positions,
                        *(ax_v1 + i + 1 for i in range(N - self._N)),
                    )
                else:
                    # Remove positions from the end, by default invisible.
                    nPositions = positions[:N]
            # Reset the handles.
            self._edge_handles.remove()
            self._edge_handles = ToolLineHandles(
                self.ax,
                nPositions,
                direction=self.direction,
                line_props=self._handle_props,
                useblit=self.useblit,
            )
            # Finally update the number of selections.
            self._N = N

    def color_spans(
        self,
        rect_colors: (
            list[str]
            | str
            | list[tuple[float, float, float, float]]
            | tuple[float, float, float, float]
            | Colormap
        ) = mpl.colormaps.get("tab10"),
        handle_colors: (
            list[str]
            | str
            | list[tuple[float, float, float, float]]
            | tuple[float, float, float, float]
            | Colormap
            | None
        ) = None,
    ) -> None:
        """
        Set the colors of the rectangle spans and their edge handles.

        Parameters
        ----------
        colors : list[str] | list[tuple[float, float, float, float]] | tuple[float, float, float, float] | str | Colormap
            A general mapping of colors to the spans.
            If a list is provided, it should match the length of N.
            If a single color is provided, all spans will be set to that color.
            Values can be a string, a tuple of RGBA values (0-1), or a Colormap.

        handle_colors : list[str] | str | list[tuple[float, float, float, float]] | tuple[float, float, float, float] | Colormap, optional
            A general mapping of colors to the edge handles. Must match the same type as colors, as behaviour will be the same.
            By default None, the edge handles will be set to half the RGB values of the rectangle colors.
        """
        rect_clist = []
        handle_clist = []
        # Check typing of handle_colors
        if handle_colors is not None:
            if type(handle_colors) != type(rect_colors):
                raise TypeError("handle_colors must match the type of rect_colors.")
            if hasattr(rect_colors, "__len__") and hasattr(handle_colors, "__len__"):
                if len(rect_colors) != len(handle_colors):
                    raise ValueError(
                        "rect_colors and handle_colors must be the same length."
                    )
                if hasattr(rect_colors[0], "__len__") and hasattr(
                    handle_colors[0], "__len__"
                ):
                    if len(rect_colors[0]) != len(handle_colors[0]):
                        raise ValueError(
                            "rect_colors and handle_colors must have the same number of elements."
                        )

        ## Gather all colors for the rectangles and edge handles.
        # LIST
        if isinstance(rect_colors, list):
            rect_clist = rect_colors.copy()
            if isinstance(rect_colors[0], str):
                handle_clist = (
                    handle_colors.copy()
                    if handle_colors is not None
                    else rect_clist.copy()
                )
            elif isinstance(rect_colors[0], (tuple, list)):
                handle_clist = (
                    # List of RGBA from rect_colors
                    [
                        (
                            (*(RGB * 0.5 for RGB in color[0:3]), color[3])
                            if len(rect_colors[0]) > 3
                            # List of RGB from rect_colors
                            else (RGB * 0.5 for RGB in color[0:3])
                        )
                        for color in rect_colors
                    ]
                    if handle_colors is None
                    # If handle_colors is defined, use that for the edge handles.
                    else handle_colors
                )
                if len(rect_colors) > 3:
                    handle_clist = [RGB + (rect_colors[3],) for RGB in handle_clist]
            else:
                raise TypeError("rect_colors must be a list of strings or tuples.")
        # TUPLE
        elif isinstance(rect_colors, tuple):
            rect_clist = [rect_colors for _ in range(self.N)]
            handle_clist = (
                # List of RGBA from rect_colors tuple
                [
                    (
                        (*(RGB * 0.5 for RGB in rect_colors[0:3]), rect_colors[3])
                        if len(rect_colors) > 3
                        # List of RGB from rect_colors tuple
                        else (RGB * 0.5 for RGB in rect_colors[0:3])
                    )
                    for _ in range(self.N)
                ]
                if handle_colors is None
                else [handle_colors for _ in range(self.N)]
            )
        # STR
        elif isinstance(rect_colors, str):
            # String cannot be used to set darker edge handles. Just copy.
            rect_clist = [rect_colors for _ in range(self.N)]
            handle_clist = (
                rect_clist.copy()
                if handle_colors is None
                else [handle_colors for _ in range(self.N)]
            )
        # CMAP
        elif isinstance(rect_colors, Colormap):
            # Use discrete colormap (i.e., mpl.colormaps.get("tab10")) to get N colours.
            rect_clist = [rect_colors(i) for i in range(self.N)]
            handle_clist = (
                # List of RGBA from rect_colors tuple
                [
                    (
                        (*(RGB * 0.5 for RGB in color[0:3]), color[3])
                        if len(color) > 3
                        # List of RGB from rect_colors tuple
                        else (RGB * 0.5 for RGB in color[0:3])
                    )
                    for color in rect_clist
                ]
                if handle_colors is None
                else [handle_colors(i) for i in range(self.N)]
            )
        else:
            raise TypeError("rect_colors must be a list, string or Colormap.")

        ## Apply colours to the rectangles and edge handles.
        for i, selection_artist in enumerate(self._selection_artists):
            selection_artist.set_facecolor(rect_clist[i])
            if self._interactive:
                self._edge_handles._artists[i * 2].set_color(handle_clist[i])
                self._edge_handles._artists[i * 2 + 1].set_color(handle_clist[i])
        return

    # --------------- Override attributes of _SelectorWidget ---------------:
    # overrides artists property
    @property
    def artists(self):
        # Overrides to include self._selection_artists instead of
        # self._selection_artist in the list of artists.
        handles_artists = getattr(self, "_handles_artists", ())
        return (*self._selection_artists,) + handles_artists

    @overrides.overrides
    def set_props(self, **props) -> None:
        # Overrides to set properties for all selection artists,
        # instead of just the single selection artist.
        artists = self._selection_artists
        for artist in artists:
            props = cbook.normalize_kwargs(props, artist)
            artist.set(**props)
        if self.useblit:
            self.update()
        # Additionally updates stored props
        self._handle_props.update(props)

    # --------------- Override attributes of SpanSelector ---------------:
    @overrides.overrides
    def _setup_edge_handles(self, props):
        ## Overrides to define 2*self.N handles for selection in _edge_handles.
        # Define initial position using the axis bounds to keep the same bounds
        if self.direction == "horizontal":
            positions = self.ax.get_xbound()
        else:
            positions = self.ax.get_ybound()

        # Use N to define 2*N handles for selection in _edge_handles.
        dxy = (positions[1] - positions[0]) / (2 * self.N - 1)
        nPositions = (positions[0] + dxy * i for i in range(2 * self.N))
        # Set handles.
        self._edge_handles = ToolLineHandles(
            self.ax,
            nPositions,
            direction=self.direction,
            line_props=props,
            useblit=self.useblit,
        )

    @overrides.overrides
    def new_axes(self, ax, *, _props=None):
        """Set SpanSelector to operate on a new Axes."""
        # Overrides to define self.N rectangles for selection in rect_artists
        # and self._selection_artists variables. Previously was rect_artist and
        # self._selection_artist.

        # Also implements an axis update, where the number of selections can be changed.

        # Reset selection on new axes.
        self._selection_completed = False
        if (
            self.ax is ax
            and hasattr(self, "_selection_artists")
            and self._selection_artists is not None
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
                    v1, v2 = self.ax.get_xbound()[0]
                    if art_len > 0:
                        v1 = (
                            self._selection_artists[-1].get_x()
                            + self._selection_artists[-1].get_width()
                        )
                    self._selection_artists += [
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
                            transform=ax.transData,
                            visible=False,
                        )
                        for i in range(self.N - art_len)
                    ]
        else:
            self.ax = ax
            # assert isinstance(self.ax, Axes) # allow editor to recognise type.
            if self.canvas is not ax.figure.canvas:
                if self.canvas is not None:
                    self.disconnect_events()

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

    @overrides.overrides
    def _press(self, event):
        """Button press event handler."""
        # Overrides to use self._selection_artists instead of self._selection_artist.
        # Also adjust extents for only the active handle.

        xdata, ydata = self._get_data_coords(event)
        v = xdata if self.direction == "horizontal" else ydata
        self._set_cursor(True)

        # Tracking for non-visible selection artists; used in self._release if the selection is less than minspan;
        # removes nearest visible instead of revealing invisible.
        self._active_handle_vis = True

        # Check if  hovering an existing visible handle.
        index, e_dist = self._edge_handles.closest(event.x, event.y)
        hover = (
            e_dist <= self.grab_range
            and self._edge_handles.artists[index].get_visible()
        )

        # If any artists rect selection artists are not visible, add one at the cursor location
        sel_artists_vis = np.array(
            [artist.get_visible() for artist in self._selection_artists], bool
        )
        if (self._visible is False or np.any(sel_artists_vis == False)) and not hover:
            i = np.argmax(sel_artists_vis == False)
            ex = self.extents
            ex[i] = v, v
            self.extents = ex

            self._active_handle_vis = False
            self.set_visible(True, i)
            # Setting active handle again when no handles are defined (ie. self._interactive = False) is necessary.
            self._active_selection_idx = i

        # Set the active handle based on the location of the mouse event.
        visible_rects = [
            select_artist.get_visible() for select_artist in self._selection_artists
        ]
        if any(visible_rects) and self._interactive:
            self._set_active_handle(event)
        elif any(visible_rects) and self._active_selection_idx is not None:
            # When not interactive, but active handle index has been defined.
            self._active_handle = None
        else:
            self._active_handle = None
            self._active_selection_idx = None

        # If no handle is active, then we are done.
        if self._active_handle is None or not self._interactive:
            # Clear previous rectangle before drawing new rectangle.
            self.update()

        print(self._active_selection_idx, self._active_handle, self._extents_on_press)

        return False

    @overrides.overrides
    def _set_active_handle(self, event):
        """Set active handle based on the location of the mouse event."""
        # Overrides to track which handle index is active.

        # Note: event.xdata/ydata in data coordinates, event.x/y in pixels
        e_idx, e_dist = self._edge_handles.closest(event.x, event.y)
        h_idx = int(np.floor(e_idx / 2))  # Handle index
        edge_order_idx = e_idx % 2  # 0 for min, 1 for max

        # Save selection index. 2 handles per selection, so divide by 2.
        self._active_selection_idx = h_idx

        # Check if another selection artist contains the event.
        # This is used after closest contains is checked.
        contains = [
            artist.contains(event) and i != h_idx and artist.get_visible()
            for i, artist in enumerate(self._selection_artists)
        ]

        ## Prioritise within proximity to edge handle, then centre handle, then outside.
        if e_dist < self.grab_range:
            # Closest to an edge handle
            self._active_handle = self._edge_order[edge_order_idx]
        elif self.drag_from_anywhere and self._contains(event):
            # Check if we've clicked inside any region.
            # Note: self._contains changes self._active_handle_idx if True for a different index.
            # Update the extents, in case the active handle index has changed.
            h_idx = self._active_selection_idx
            self._active_handle = "C"
        elif "move" in self._state:
            self._active_handle = "C"
        elif not self.ignore_event_outside:
            # If outside the region, instead use closest edge determine which selection.
            self._active_handle = self._edge_order[edge_order_idx]
            # Get the extents for the active handle.
            xdata, ydata = self._get_data_coords(event)
            v = xdata if self.direction == "horizontal" else ydata
            ex = self.extents
            ex[h_idx] = v, v
            self.extents = ex
        else:
            self._active_handle = None
            self._active_selection_idx = None

        # Save coordinates of rectangle at the start of handle movement.
        self._extents_on_press = self.extents[h_idx]

    @overrides.overrides
    def _contains(self, event):
        """Return True if event is within the selected handle index."""
        # Note, if the event is within another span, then the active handle index is updated.
        i = self._active_selection_idx
        if i is None:
            return False
        else:
            if self._selection_artists[i].contains(event, radius=0)[0]:
                return True
            else:
                # Loop through other selection artists to check if event is within any.
                for j, selection_artist in enumerate(self._selection_artists):
                    if j != i and selection_artist.contains(event, radius=0)[0]:
                        # Update active handle index to the selection artist that contains the event.
                        self._active_selection_idx = j
                        return True

    @overrides.overrides
    def _onmove(self, event):
        """Motion notify event handler."""

        xdata, ydata = self._get_data_coords(event)
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

    @overrides.overrides
    def _hover(self, event):
        """Update the canvas cursor if it's over a handle."""
        # Overrides behaviour so that hovering edge_handles will work with incomplete selections.
        # But ignore invisible selections.
        if self.ignore(event):
            return

        if self._active_handle is not None:
            # Do nothing if button is pressed and a handle is active, which may
            # occur with drag_from_anywhere=True.
            return

        index, e_dist = self._edge_handles.closest(event.x, event.y)
        self._set_cursor(
            e_dist <= self.grab_range
            and self._edge_handles.artists[index].get_visible()
        )

    @overrides.overrides
    def set_visible(self, visible: bool, index: int = None) -> None:
        """
        Overrides functionality of default _SelectorWidget.set_visible method.
        Uses additional artists and sets visibility for all selection artists if no index is provided.

        Parameters
        ----------
        visible : bool
            The visibility of the selection artists.
        index : int, optional
            The index of the selection artist to set visibility for. If None, then
            all selection artists will be set to the same visibility.
        """
        if index is not None:
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

    @overrides.overrides
    def _release(self, event):
        """Button release event handler."""
        self._set_cursor(False)

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
        # If not interactive (ie, spans disappear after full selection) then hide once the last selection is made.
        if not self._interactive:
            # Check if the active handle belongs to the last required selection (last two handles).
            if (
                self._active_selection_idx is not None
                and self._active_selection_idx >= self.N - 1
            ):
                for artist in self._selection_artists:
                    artist.set_visible(False)

        # Hide current span in case the span is less than minspan.
        # Perform again later if changing handle
        idx = self._active_selection_idx
        if self._active_selection_idx is not None:
            if spans[idx] <= self.minspan:
                self.set_visible(False, index=idx)

        # Find visible selection artists, exclusing the active handle.
        visible_rects = [
            select_artist
            for i, select_artist in enumerate(self._selection_artists)
            if select_artist.get_visible() and i != idx
        ]
        # Check if active selection wasn't and won't be visible
        if (
            not self._active_handle_vis
            and idx
            and spans[idx] <= self.minspan
            and len(visible_rects) > 0
        ):
            # Hide the nearest visible selection if any visible.
            # Use release x,y to find closest visible.
            xdata, ydata = self._get_data_coords(event)
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
                # If callback only at selection complete, do not call as selection is incomplete.
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
                                        f"onselect `{self.onselect[i]}` must be a callable."
                                    )
                            else:
                                if callable(self.onselect):
                                    self.onselect(wmin, wmax)
                                else:
                                    raise ValueError(
                                        f"onselect `{self.onselect}` must be a callable."
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
            # Do not call for incomplete selections, or hidden selections remaining hidden.
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

    @property
    def extents(self) -> list[tuple[float, float]]:
        """
        Returns
        -------
        list[tuple[float, float]]
            The values, in data coordinates, for the start and end points of all current rectangles.
            If there is no selection then the start and end values will be the same.
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
        self._draw_shapes(extents)
        if self._interactive:
            # Update displayed handles
            extent_data = np.array(extents).flatten()
            self._edge_handles.set_data(extent_data)
        self.update()

    def _draw_shapes(
        self,
        extents: list[tuple[float, float]],
    ):
        """An alternative method for SpanSelector._draw_shape.
        Draws the selection shapes on the axes."""
        for i, extent in enumerate(extents):
            (vmin, vmax) = extent
            # Reorder if necessary
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            if self.direction == "horizontal":
                self._selection_artists[i].set_x(vmin)
                self._selection_artists[i].set_width(vmax - vmin)
            else:
                self._selection_artists[i].set_y(vmin)
                self._selection_artists[i].set_height(vmax - vmin)

    @property
    def colors_rect(self) -> list[str]:
        """
        Property to get/set the colours of the rectangle spans.

        Parameters
        ----------
        colors : list[str]
            List of colours to set for each rectangle span.

        Returns
        -------
        list[str]
            List of colours for each rectangle
        """
        return [artist.get_facecolor() for artist in self._selection_artists]

    @colors_rect.setter
    def colors_rect(self, colors: list[str] | str):
        if isinstance(colors, list):
            for i, color in enumerate(colors):
                if len(color) > 3:
                    self._selection_artists[i].set_alpha(color[3])
                self._selection_artists[i].set_facecolor(color[0:3])
        elif isinstance(colors, str):
            for artist in self._selection_artists:
                if len(color) > 3:
                    artist.set_alpha(color[3])
                artist.set_facecolor(color[0:3])
        else:
            raise ValueError(f"colors '{colors}' must be a list or a string")
        self.update()

    @property
    def colors_edges(self) -> list[tuple[str, str]]:
        """
        Property to get the colours of the edge handles.

        Parameters
        ----------
        colors : list[tuple[str,str]]
            List of colours to set for each edge handle.

        Returns
        -------
        list[tuple[str,str]]
            List of colours for each edge handle
        """
        colors = [artist.get_color() for artist in self._edge_handles.artists]
        colors = [(colors[i], colors[i + 1]) for i in range(0, len(colors), 2)]
        return colors

    @colors_edges.setter
    def colors_edges(self, colors: list[tuple[str, str]] | tuple[str, str] | str):
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
        pass

    def onselect2(xmin, xmax):
        print("2nd Span:\t", xmin, xmax)
        pass

    def onselect3(xmin, xmax):
        print("3rd Span:\t", xmin, xmax)
        thecols = span.colors_rect
        newGreen = np.sqrt(thecols[-1])
        newGreen[0:3] = thecols[-1][0:3]  # keep colour, change alpha
        thecols[-1] = tuple(newGreen)
        span.colors_rect = thecols
        pass

    span = NSpanSelector(
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
