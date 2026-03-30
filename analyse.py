#!/usr/bin/env python
# coding: utf-8

# In[1]:


def generate_dashboard():
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.io as pio
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    import webbrowser
    import os

    from datetime import datetime
    import pytz
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # In[2]:

    nltk.download("vader_lexicon", quiet=True)

    # # Loading Data

    # In[3]:

    apps_df = pd.read_csv("play-store-data.csv")
    reviews_df = pd.read_csv("user-reviews.csv")

    # # View DataType

    # In[4]:

    apps_df.dtypes

    # In[5]:

    reviews_df.dtypes

    # # Cleaning the Data

    # In[6]:

    apps_df = apps_df.dropna(subset=["Rating"])
    for column in apps_df.columns:
        apps_df[column] = apps_df[column].fillna(apps_df[column].mode()[0])
    apps_df.drop_duplicates(inplace=True)
    apps_df = apps_df[apps_df["Rating"] <= 5]

    reviews_df.dropna(subset=["Translated_Review"], inplace=True)

    # #### Converts Install column to integer by removing commas and '+'

    # In[7]:

    apps_df["Installs"] = (
        apps_df["Installs"].str.replace(",", "").str.replace("+", "").astype(int)
    )

    # #### Converts Price column to Integer by removing ''$''

    # In[8]:

    apps_df["Price"] = apps_df["Price"].str.replace("$", "").astype(float)

    # In[9]:

    apps_df["Installs"] = apps_df["Installs"].astype(int)
    apps_df["Reviews"] = apps_df["Reviews"].astype(int)

    # In[10]:

    merged_df = pd.merge(apps_df, reviews_df, on="App", how="inner")

    # # Data Transformation

    # In[11]:

    def convert_size(size):
        if "M" in size:
            return float(size.replace("M", ""))
        elif "k" in size:
            return float(size.replace("k", "")) / 1024
        else:
            return np.nan

    apps_df["Size"] = apps_df["Size"].apply(convert_size)

    # In[12]:

    apps_df

    # In[13]:

    apps_df["Log_Installs"] = np.log1p(apps_df["Installs"])
    apps_df["Log_Reviews"] = np.log1p(apps_df["Reviews"])

    # In[14]:

    def rating_group(rating):
        if rating >= 4:
            return "Top rated app"
        if rating >= 3:
            return "Above average"
        if rating >= 2:
            return "Average"
        return "Below Average"

    # In[15]:

    apps_df["Rating_group"] = apps_df["Rating"].apply(rating_group)

    # **Revenue Column**

    # In[16]:

    apps_df["Revenue"] = apps_df["Price"] * apps_df["Installs"]

    # In[17]:

    sia = SentimentIntensityAnalyzer()

    # # Polarity Scores in SIA

    # In[18]:

    reviews_df["Sentiment_Scores"] = reviews_df["Translated_Review"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    # In[19]:

    apps_df["Last Updated"] = pd.to_datetime(apps_df["Last Updated"], errors="coerce")

    # In[20]:

    apps_df["Year"] = apps_df["Last Updated"].dt.year

    # # Plotly

    # In[21]:

    html_files_path = "./"
    if not os.path.exists(html_files_path):
        os.makedirs(html_files_path)

    # In[22]:

    plot_containers = """"""

    # In[23]:

    def save_plot_as_html(fig, filename, insight):
        nonlocal plot_containers
        file_path = os.path.join(html_files_path, filename)
        html_content = pio.to_html(fig, full_html=False, include_plotlyjs="inline")
        plot_containers += f"""
        <div class = "plot-container" id = "{filename}" onclick = "openPlot('{filename}')">
            <div class = "plot"> {html_content} </div>
            <div class = "insights">{insight}</div>
        </div>
        """

    # In[24]:

    plot_width = 400
    plot_height = 300
    plot_bg_color = "black"
    text_color = "white"
    title_font = {"size": 16}
    axis_font = {"size": 12}

    # ***Figure 1***

    # In[25]:

    category_counts = apps_df["Category"].value_counts().nlargest(10)
    fig1 = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        labels={"x": "Category", "y": "Count"},
        title="Top Categories on PlayStore",
        color=category_counts.index,
        color_discrete_sequence=px.colors.sequential.Plasma,
        width=400,
        height=300,
    )

    fig1.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        title_font={"size": 16},
        xaxis=dict(title_font={"size": 12}),
        yaxis=dict(title_font={"size": 12}),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    # fig1.update_traces(marker = dict(marker = dict(line = dict(color = "white" , width = 1))))
    save_plot_as_html(
        fig1, "Category Graph 1.html", "The Top Categories apps on Playstore"
    )

    # ***Figure 2***

    # In[26]:

    type_counts = apps_df["Type"].value_counts()
    fig2 = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title="App type distribution",
        color_discrete_sequence=px.colors.sequential.RdBu,
        width=400,
        height=300,
    )

    fig2.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        title_font={"size": 16},
        margin=dict(l=10, r=10, t=30, b=10),
    )

    # fig1.update_traces(marker = dict(marker = dict(line = dict(color = "white" , width = 1))))
    save_plot_as_html(
        fig2,
        "Type Graph 1.html",
        "Most apps in the PlayStore are free, indicating a strategy to attract users first",
    )

    # ***Figure 3***

    # In[27]:

    fig3 = px.histogram(
        apps_df,
        x="Rating",
        nbins=20,
        title="Rating distribution",
        color_discrete_sequence=["#636EFA"],
        width=400,
        height=300,
    )

    fig3.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        title_font={"size": 16},
        xaxis=dict(title_font={"size": 12}),
        yaxis=dict(title_font={"size": 12}),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    # fig1.update_traces(marker = dict(marker = dict(line = dict(color = "white" , width = 1))))
    save_plot_as_html(
        fig3,
        "Rating Graph 3.html",
        "Ratings are skewed towards higher values, suggesting that most apps are suggested favorably by users",
    )

    # ***Figure 4***

    # In[28]:

    sentiment_counts = reviews_df["Sentiment_Scores"].value_counts()
    fig4 = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        labels={"x": "Sentiment Score", "y": "Count"},
        title="Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_sequence=px.colors.sequential.RdPu,
        width=400,
        height=300,
    )

    fig4.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        title_font={"size": 16},
        xaxis=dict(title_font={"size": 12}),
        yaxis=dict(title_font={"size": 12}),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    # fig1.update_traces(marker = dict(marker = dict(line = dict(color = "white" , width = 1))))
    save_plot_as_html(
        fig4,
        "Sentiment Graph 4.html",
        "Sentiment in reviews shows mixed positive and negative feedback",
    )

    # ***Figure 5***

    # In[29]:

    installs_by_category = apps_df.groupby("Category")["Installs"].sum().nlargest(10)
    fig5 = px.bar(
        x=installs_by_category.index,
        y=installs_by_category.values,
        orientation="h",
        labels={"x": "Installs", "y": "Category"},
        title="Installs by Category",
        color=installs_by_category.index,
        color_discrete_sequence=px.colors.sequential.Blues,
        width=400,
        height=300,
    )

    fig5.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        title_font={"size": 16},
        xaxis=dict(title_font={"size": 12}),
        yaxis=dict(title_font={"size": 12}),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    # fig1.update_traces(marker = dict(marker = dict(line = dict(color = "white" , width = 1))))
    save_plot_as_html(
        fig5,
        "Installs Graph 5.html",
        "Categories with most Installs are Social and communication apps , reflecting their broad usage",
    )

    # ***Figure 6***

    # In[30]:

    updates_per_year = apps_df["Last Updated"].dt.year.value_counts().sort_index()
    fig6 = px.line(
        x=updates_per_year.index,
        y=updates_per_year.values,
        labels={"x": "Year", "y": "No of Updates"},
        title="Number Of Updates Over The Years",
        color_discrete_sequence=["#AB63FA"],
        width=400,
        height=300,
    )

    fig6.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        title_font={"size": 16},
        xaxis=dict(title_font={"size": 12}),
        yaxis=dict(title_font={"size": 12}),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    # fig1.update_traces(marker = dict(marker = dict(line = dict(color = "white" , width = 1))))
    save_plot_as_html(
        fig6,
        "Updates Graph 6.html",
        "Updates have been increasing over the year, showing that developers are actively maintaining and improving their apps",
    )

    # ***Figure 7***

    # In[31]:

    revenue_by_category = apps_df.groupby("Category")["Revenue"].sum().nlargest(10)
    fig7 = px.bar(
        x=installs_by_category.index,
        y=installs_by_category.values,
        labels={"x": "Category", "y": "Revenue"},
        title="Revenue by Category",
        color=installs_by_category.index,
        color_discrete_sequence=px.colors.sequential.Greens,
        width=400,
        height=300,
    )

    fig7.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        title_font={"size": 16},
        xaxis=dict(title_font={"size": 12}),
        yaxis=dict(title_font={"size": 12}),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    # fig1.update_traces(marker = dict(marker = dict(line = dict(color = "white" , width = 1))))
    save_plot_as_html(
        fig7,
        "Revenue Graph 7.html",
        "Categories such as Business and Productivity leads in revenue",
    )

    # ***Figure 8***

    # In[32]:

    genre_counts = (
        apps_df["Category"]
        .str.split(";", expand=True)
        .stack()
        .value_counts()
        .nlargest(10)
    )
    fig8 = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        labels={"x": "Genre", "y": "Count"},
        title="Top Genres",
        color=installs_by_category.index,
        color_discrete_sequence=px.colors.sequential.OrRd,
        width=400,
        height=300,
    )

    fig8.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        title_font={"size": 16},
        xaxis=dict(title_font={"size": 12}),
        yaxis=dict(title_font={"size": 12}),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    # fig1.update_traces(marker = dict(marker = dict(line = dict(color = "white" , width = 1))))
    save_plot_as_html(
        fig8, "Genre Graph 8.html", "Actions and Casual genres are the most common "
    )

    # ***Figure 9***

    # In[33]:

    fig9 = px.scatter(
        apps_df,
        x="Last Updated",
        y="Rating",
        color="Type",
        title="Impact of Last Update on Rating",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        width=400,
        height=300,
    )
    fig9.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        title_font={"size": 16},
        xaxis=dict(title_font={"size": 12}),
        yaxis=dict(title_font={"size": 12}),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    # fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
    save_plot_as_html(
        fig9,
        "Update Graph 9.html",
        "The Scatter Plot shows a weak correlation between the last update and ratings, suggesting that more frequent updates dont always result in better ratings.",
    )

    # ***Figure 10***

    # In[34]:

    # Figure 10
    fig10 = px.box(
        apps_df,
        x="Type",
        y="Rating",
        color="Type",
        title="Rating for Paid vs Free Apps",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        width=400,
        height=300,
    )
    fig10.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        title_font={"size": 16},
        xaxis=dict(title_font={"size": 12}),
        yaxis=dict(title_font={"size": 12}),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    # fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
    save_plot_as_html(
        fig10,
        "Paid Free Graph 10.html",
        "Paid apps generally have higher ratings compared to free apps, suggesting that users expect higher quality from apps they pay for",
    )

    # # INTERNSHIP TASKS

    # ## Task 1

    # In[35]:

    IST = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(IST)
    current_hour = now_ist.hour

    if not (17 <= current_hour < 19):
        print(
            f"The bubble chart is hidden. It only appears between 5 PM and 7 PM IST. (Current IST: {now_ist.strftime('%H:%M')})"
        )
    else:
        avg_subjectivity = (
            reviews_df.groupby("App")["Sentiment_Subjectivity"].mean().reset_index()
        )

        bubble_df = pd.merge(apps_df, avg_subjectivity, on="App", how="inner")

        target_categories = [
            "GAME",
            "BEAUTY",
            "BUSINESS",
            "COMICS",
            "COMMUNICATION",
            "DATING",
            "ENTERTAINMENT",
            "SOCIAL",
            "EVENTS",
        ]

        translations = {
            "BEAUTY": "सौंदर्य",  # Hindi
            "BUSINESS": "வணிகம்",  # Tamil
            "DATING": "Dating",  # German (common usage)
        }

        filtered_bubble = bubble_df[
            (bubble_df["Rating"] > 3.5)
            & (bubble_df["Category"].isin(target_categories))
            & (bubble_df["Reviews"] > 500)
            & (~bubble_df["App"].str.contains("S", case=True, na=False))
            & (bubble_df["Sentiment_Subjectivity"] > 0.5)
            & (bubble_df["Installs"] > 50000)
        ].copy()

        filtered_bubble["Display_Category"] = filtered_bubble["Category"].replace(
            translations
        )

        unique_cats = filtered_bubble["Display_Category"].unique()
        color_map = {}
        standard_palette = px.colors.qualitative.Safe

        for i, cat in enumerate(unique_cats):
            if cat == "GAME":
                color_map[cat] = "#FF69B4"  # Hot Pink
            else:
                color_map[cat] = standard_palette[i % len(standard_palette)]

        # 8. Plot Bubble Chart
        fig11 = px.scatter(
            filtered_bubble,
            x="Size",
            y="Rating",
            size="Installs",
            color="Display_Category",
            hover_name="App",
            color_discrete_map=color_map,
            size_max=60,
            title="App Size vs Average Rating",
            labels={
                "Size": "App Size (MB)",
                "Rating": "Average Rating",
                "Display_Category": "Category",
            },
            width=400,
            height=300,
        )

        fig11.update_layout(
            plot_bgcolor="black",
            paper_bgcolor="black",
            font_color="white",
            title_font={"size": 16},
            margin=dict(l=10, r=10, t=50, b=10),
        )

        # 9. Save to Dashboard
        save_plot_as_html(
            fig11,
            "Bubble Chart 11.html",
            "Relationship between App Size and Rating for specific categories (Filtered for Rating > 3.5, Reviews > 500, Installs > 50k, and high Sentiment Subjectivity).",
        )
        print("  Bubble chart successfully generated and added to dashboard.")

    # ## Task 2

    # In[36]:

    IST = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(IST)
    current_hour = now_ist.hour

    if not (18 <= current_hour < 20):
        print(
            f"  Task 2 (Choropleth Map) is hidden. It only appears between 6 PM and 8 PM IST. (Current IST: {now_ist.strftime('%I:%M %p')})"
        )
    else:
        excluded_prefixes = ("A", "C", "G", "S")
        filtered_df = apps_df[
            ~apps_df["Category"].str.startswith(excluded_prefixes, na=False)
        ].copy()

        category_installs = (
            filtered_df.groupby("Category")["Installs"].sum().reset_index()
        )

        top_5_categories = category_installs.nlargest(5, "Installs").copy()

        top_5_categories["Highlight"] = top_5_categories["Installs"] > 1000000

        countries = ["USA", "IND", "GBR", "CAN", "AUS"]
        top_5_categories["Country_Code"] = countries[: len(top_5_categories)]

        fig12 = px.choropleth(
            top_5_categories,
            locations="Country_Code",
            color="Installs",
            hover_name="Category",
            color_continuous_scale=(
                "Viridis" if top_5_categories["Highlight"].any() else "Greens"
            ),
            title="Global Installs by Category",
            labels={
                "Installs": "Total Installs",
                "Country_Code": "Representative Country",
            },
            width=400,
            height=300,
        )

        fig12.update_layout(
            plot_bgcolor="black",
            paper_bgcolor="black",
            font_color="white",
            title_font={"size": 16},
            margin=dict(l=20, r=20, t=50, b=20),
            geo=dict(
                bgcolor="black",
                showframe=False,
                showcoastlines=True,
                projection_type="equirectangular",
            ),
        )

        save_plot_as_html(
            fig12,
            "Global_Installs_Map_12.html",
            "Global installs visualized by top 5 categories (excluding A, C, G, S). "
            "Categories with >1M installs are represented with intense color values. "
            "Mapped to dummy countries to fulfill Choropleth requirement.",
        )
        print(
            f"  Choropleth map generated successfully at {now_ist.strftime('%I:%M %p')} IST."
        )

    # ## Task 3

    # In[37]:

    IST = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(IST)
    current_hour = now_ist.hour

    if not (18 <= current_hour < 21):
        print(
            f"  Task 3 is hidden. It only appears between 6 PM and 9 PM IST. (Current IST: {now_ist.strftime('%I:%M %p')})"
        )
    else:
        df_ts = apps_df.copy()

        df_ts["Last Updated"] = pd.to_datetime(df_ts["Last Updated"], errors="coerce")
        df_ts = df_ts.dropna(subset=["Last Updated"])

        df_ts = df_ts[~df_ts["App"].str.lower().str.startswith(("x", "y", "z"))]

        df_ts = df_ts[~df_ts["App"].str.contains("s", case=False, na=False)]

        df_ts = df_ts[df_ts["Category"].str.startswith(("E", "C", "B"), na=False)]

        # Filter 4: Reviews > 500
        df_ts = df_ts[df_ts["Reviews"] > 500]

        translations = {
            "BEAUTY": "सौंदर्य",  # Hindi
            "BUSINESS": "வணிகம்",  # Tamil
            "DATING": "Dating",  # German
        }
        df_ts["Category"] = df_ts["Category"].replace(translations)

        df_ts["Month_Year"] = df_ts["Last Updated"].dt.to_period("M").dt.to_timestamp()

        trend_df = (
            df_ts.groupby(["Category", "Month_Year"])["Installs"].sum().reset_index()
        )
        trend_df = trend_df.sort_values(["Category", "Month_Year"])

        trend_df["MoM_Growth"] = trend_df.groupby("Category")["Installs"].pct_change()

        fig13 = px.line(
            trend_df,
            x="Month_Year",
            y="Installs",
            color="Category",
            title="Trend of Total Installs",
            labels={"Month_Year": "Date", "Installs": "Total Installs"},
            template="plotly_dark",
        )

        growth_periods = trend_df[trend_df["MoM_Growth"] > 0.20]

        for _, row in growth_periods.iterrows():
            fig13.add_vrect(
                x0=row["Month_Year"],
                x1=row["Month_Year"] + pd.DateOffset(months=1),
                fillcolor="green",
                opacity=0.1,
                layer="below",
                line_width=0,
            )

        fig13.update_layout(
            plot_bgcolor="black",
            paper_bgcolor="black",
            font_color="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#333"),
            width=400,
            height=300,
        )

        save_plot_as_html(
            fig13,
            "Install_Trend_13.html",
            "Trend of installs for categories starting with E, C, and B. "
            "Green shaded areas indicate >20% Month-over-Month growth. "
            "Filtered to exclude apps containing 'S' or starting with X, Y, Z.",
        )
        print("  Task 3: Time series chart successfully generated.")

    # ## Task 4

    # In[38]:

    IST = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(IST)
    current_hour = now_ist.hour

    if not (16 <= current_hour < 18):
        print(
            f"  Task 4 (Stacked Area Chart) is hidden. It only appears between 4 PM and 6 PM IST. (Current IST: {now_ist.strftime('%I:%M %p')})"
        )
    else:
        df_area = apps_df.copy()

        df_area["Last Updated"] = pd.to_datetime(
            df_area["Last Updated"], errors="coerce"
        )
        df_area = df_area.dropna(subset=["Last Updated"])

        df_area = df_area[df_area["Rating"] >= 4.2]

        df_area = df_area[~df_area["App"].str.contains(r"\d", na=False)]

        df_area = df_area[df_area["Category"].str.startswith(("T", "P"), na=False)]

        df_area = df_area[df_area["Reviews"] > 1000]

        df_area = df_area[(df_area["Size"] >= 20) & (df_area["Size"] <= 80)]

        category_translations = {
            "TRAVEL_AND_LOCAL": "Voyages et local",  # French
            "PRODUCTIVITY": "Productividad",  # Spanish
            "PHOTOGRAPHY": "写真",  # Japanese (Shashin)
        }
        df_area["Category_Display"] = df_area["Category"].replace(category_translations)

        df_area["Month"] = df_area["Last Updated"].dt.to_period("M").dt.to_timestamp()
        area_plot_df = (
            df_area.groupby(["Month", "Category_Display"])["Installs"]
            .sum()
            .reset_index()
        )
        area_plot_df = area_plot_df.sort_values(["Category_Display", "Month"])

        area_plot_df["Cumulative_Installs"] = area_plot_df.groupby("Category_Display")[
            "Installs"
        ].cumsum()

        area_plot_df["MoM_Growth"] = area_plot_df.groupby("Category_Display")[
            "Installs"
        ].pct_change()

        fig14 = px.area(
            area_plot_df,
            x="Month",
            y="Cumulative_Installs",
            color="Category_Display",
            title="Cumulative Installs Over Time",
            labels={
                "Month": "Date",
                "Cumulative_Installs": "Cumulative Installs",
                "Category_Display": "App Category",
            },
            template="plotly_dark",
        )

        growth_highlights = area_plot_df[area_plot_df["MoM_Growth"] > 0.25]

        for _, row in growth_highlights.iterrows():
            fig14.add_vrect(
                x0=row["Month"],
                x1=row["Month"]
                + pd.DateOffset(days=20),  # Shade the vicinity of the growth month
                fillcolor="white",
                opacity=0.15,
                layer="below",
                line_width=0,
            )

        fig14.update_layout(
            plot_bgcolor="black",
            paper_bgcolor="black",
            font_color="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#444"),
            width=400,
            height=300,
        )

        save_plot_as_html(
            fig14,
            "Stacked_Area_Chart_14.html",
            "Cumulative installs for 'T' and 'P' categories. "
            "Legend includes translations for French, Spanish, and Japanese. "
            "White vertical bands highlight months with >25% MoM install growth.",
        )
        print(
            "  Task 4: Stacked Area Chart successfully generated and added to the dashboard."
        )

    # ## Task 5

    # In[39]:

    IST = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(IST)
    current_hour = now_ist.hour

    if not (15 <= current_hour < 17):
        print(
            f"  Task 5 (Grouped Bar Chart) is hidden. It only appears between 3 PM and 5 PM IST. (Current IST: {now_ist.strftime('%I:%M %p')})"
        )
    else:
        df_task5 = apps_df.copy()

        df_task5["Last Updated"] = pd.to_datetime(
            df_task5["Last Updated"], errors="coerce"
        )

        df_task5 = df_task5[
            (df_task5["Size"] >= 10) & (df_task5["Last Updated"].dt.month == 1)
        ]

        cat_agg = (
            df_task5.groupby("Category")
            .agg({"Installs": "sum", "Rating": "mean", "Reviews": "sum"})
            .reset_index()
        )

        cat_agg = cat_agg[cat_agg["Rating"] >= 4.0]

        top_10_cats = cat_agg.nlargest(10, "Installs")

        fig15 = make_subplots(specs=[[{"secondary_y": True}]])

        fig15.add_trace(
            go.Bar(
                x=top_10_cats["Category"],
                y=top_10_cats["Rating"],
                name="Avg Rating",
                marker_color="#636EFA",
            ),
            secondary_y=False,
        )

        fig15.add_trace(
            go.Bar(
                x=top_10_cats["Category"],
                y=top_10_cats["Reviews"],
                name="Total Reviews",
                marker_color="#EF553B",
            ),
            secondary_y=True,
        )

        fig15.update_layout(
            title_text="Top 10 Categories: Avg Rating vs Total Reviews (Filtered)",
            barmode="group",
            plot_bgcolor="black",
            paper_bgcolor="black",
            font_color="white",
            xaxis=dict(title="Category", tickangle=45),
            yaxis=dict(title="Average Rating (0-5)", range=[0, 5], showgrid=False),
            yaxis2=dict(title="Total Review Count", showgrid=True, gridcolor="#333"),
            width=400,
            height=300,
            legend=dict(x=1.1, y=1),
        )

        save_plot_as_html(
            fig15,
            "Grouped_Bar_Chart_15.html",
            "Comparison of Avg Rating and Reviews for top categories (by installs). "
            "Filters: App size ≥ 10MB, Updated in January, Category Avg Rating ≥ 4.0.",
        )
        print("  Task 5: Grouped Bar Chart generated and added to dashboard.")

    # ## Task 6

    # In[5]:

    IST = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(IST)
    current_hour = now_ist.hour

    if not (13 <= current_hour < 14):
        print(
            f"  Task 6 (Dual-Axis Chart) is hidden. It only appears between 1 PM and 2 PM IST. (Current IST: {now_ist.strftime('%I:%M %p')})"
        )
    else:
        df_task6 = apps_df.copy()

        df_task6["Ver_Num"] = pd.to_numeric(
            df_task6["Android Ver"].str.extract(r"(\d+\.\d+)")[0], errors="coerce"
        ).fillna(0)

        mask = (
            ~((df_task6["Installs"] < 10000) & (df_task6["Revenue"] < 10000))
            & (df_task6["Ver_Num"] > 4.0)
            & (df_task6["Size"] > 15)
            & (df_task6["Content Rating"] == "Everyone")
            & (df_task6["App"].str.len() <= 30)
        )
        filtered_6 = df_task6[mask].copy()

        top_3_cats = (
            filtered_6.groupby("Category")["Installs"].sum().nlargest(3).index.tolist()
        )
        final_df = filtered_6[filtered_6["Category"].isin(top_3_cats)]

        comparison_df = (
            final_df.groupby(["Category", "Type"])
            .agg({"Installs": "mean", "Revenue": "mean"})
            .reset_index()
        )

        fig16 = go.Figure()

        comparison_df["X_Label"] = (
            comparison_df["Category"] + " (" + comparison_df["Type"] + ")"
        )

        fig16.add_trace(
            go.Bar(
                x=comparison_df["X_Label"],
                y=comparison_df["Installs"],
                name="Avg Installs",
                marker_color="skyblue",
                yaxis="y1",
            )
        )

        fig16.add_trace(
            go.Scatter(
                x=comparison_df["X_Label"],
                y=comparison_df["Revenue"],
                name="Avg Revenue ($)",
                mode="lines+markers",
                marker=dict(size=10, color="gold"),
                line=dict(width=3),
                yaxis="y2",
            )
        )

        fig16.update_layout(
            title="Avg Installs vs Revenue: Free vs Paid (Top 3 Categories)",
            plot_bgcolor="black",
            paper_bgcolor="black",
            font_color="white",
            xaxis=dict(title="Category & App Type", tickangle=15),
            yaxis=dict(title="Average Installs", showgrid=False),
            yaxis2=dict(
                title="Average Revenue ($)",
                overlaying="y",
                side="right",
                showgrid=True,
                gridcolor="#333",
            ),
            legend=dict(x=1.1, y=1),
            width=400,
            height=300,
        )

        save_plot_as_html(
            fig16,
            "Dual_Axis_Chart_16.html",
            "Comparison of Free vs Paid apps in the top 3 categories. "
            "Filters: Installs/Revenue threshold, Android > 4.0, Size > 15MB, "
            "Content Rating: Everyone, and Name length ≤ 30 chars.",
        )
        print(
            "  Task 6: Dual-axis chart successfully generated and added to the dashboard."
        )

    # In[41]:

    plot_containers_split = plot_containers.split("</div>")

    if len(plot_containers_split) > 1:
        final_plot = plot_containers_split[-2] + "</div>"
    else:
        final_plot = plot_containers

    dashboard_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name=viewport" content="width=device-width,initial-scale-1.0">
        <title> Google Play Store Review Analytics</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #333;
                color: #fff;
                margin: 0;
                padding: 0;
            }}
            .header {{
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                background-color: #444
            }}
            .header img {{
                margin: 0 10px;
                height: 50px;
            }}
            .container {{
                display: flex;
                flex-wrap: wrap;
                justify_content: center;
                padding: 20px;
            }}
            .plot-container {{
                border: 2px solid #555
                margin: 10px;
                padding: 10px;
                width: {plot_width}px;
                height: {plot_height}px;
                overflow: hidden;
                position: relative;
                cursor: pointer;
            }}
            .insights {{
                display: none;
                position: absolute;
                right: 10px;
                top: 10px;
                background-color: rgba(0,0,0,0.7);
                padding: 5px;
                border-radius: 5px;
                color: #fff;
            }}
            .plot-container: hover .insights {{
                display: block;
            }}
            </style>
            <script>
                function openPlot(filename) {{
                    window.open(filename, '_blank');
                    }}
            </script>
        </head>
        <body>
            <div class= "header">
                <img src="/static/google_logo.webp" alt="Google Logo">
                <h1>Google Play Store Reviews Analytics</h1>
                <img src="/static/playstore_logo.webp" alt="Google Play Store Logo">
            </div>
            <div class="container">
                {plots}
            </div>
        </body>
        </html>
        """

    final_html = dashboard_html.format(
        plots=plot_containers, plot_width=plot_width, plot_height=plot_height
    )

    return final_html

    # In[ ]:
