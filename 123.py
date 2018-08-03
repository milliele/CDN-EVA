# -*- coding: utf-8 -*-

from scenario import Evaluation
from mystatus import myplot

eva = Evaluation()
eva.dynamic2()
myplot("plot")

# matplotlib.pyplot.text(x, y, s, fontdict=None, withdash=False, **kwargs)[source]
# Add text to the axes.
#
# Add the text s to the axes at location x, y in data coordinates.
#
# Parameters:
# x, y : scalars
# The position to place the text. By default, this is in data coordinates. The coordinate system can be changed using the transform parameter.
#
# s : str
# The text.
#
# fontdict : dictionary, optional, default: None
# A dictionary to override the default text properties. If fontdict is None, the defaults are determined by your rc parameters.
#
# withdash : boolean, optional, default: False
# Creates a TextWithDash instance instead of a Text instance.
#
# Returns:
# text : Text
# The created Text instance.
#
# Other Parameters:
# **kwargs : Text properties.
# Other miscellaneous text parameters.
#
# Examples





