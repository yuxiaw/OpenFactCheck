import streamlit as st
from streamlit_theme import st_theme

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):
    # Determine if the theme background is light or dark
    theme = st_theme()

    # If theme is None, set the base theme to dark
    if theme is None:
        # Wait for the theme to be set
        theme_base = "dark"
    else:
        theme_base = theme.get("base")
    
    # Set the text color based on the theme
    text_color = "white" if theme_base in ["#000000", "dark"] else "black"

    style = f"""
    <style>
      #MainMenu {{visibility: hidden;}}
      footer {{visibility: hidden;}}
     .stApp {{ bottom: 80px; }}
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color=text_color,  # Dynamic text color based on the theme
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, 8, 8),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(style=style_hr),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer(args):
    layout(*args)
