from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum
import matplotlib.pyplot as plt
import os
import pyspark.sql.functions as F
from pyspark.sql.functions import collect_list


file_path = "../data/Absence_3term201819_nat_reg_la_sch.csv" # Path to the raw dataset
parquet_file_path = "../data/absence_data.parquet" # Path to the parquet file

def read_in_dataset(spark):
    # Read in the dataset from the CSV file
    df = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(file_path)

    df = data_preparation(df)

    # Write the DataFrame to a parquet file
    df.write \
        .mode("overwrite") \
        .parquet(parquet_file_path)
    
def data_preparation(df):
    # Deal with missing data and data types

    # Replace missing valeus with Null
    df = df.replace(" ", None)

    # Convert columns to integer where necessary
    df = df.withColumn("time_period", col("time_period").cast("int"))
    
    return df

def search_enrolement_by_local_authority(df, local_authority):

    local_authorities = local_authority.split(",")
    local_authorities = [x.strip() for x in local_authorities]

    for local_authority_index in local_authorities:
        # Filter by the provided local authority and group by time period and la_name
        result = df.filter((col("geographic_level") == "Local authority") &
                   (col("la_name") == local_authority_index) &
                   (col("school_type") == "Total")) \
                .select("time_period", "la_name", "enrolments")
        result.show(truncate=False)


def search_authorised_absences_by_school_type(df, school_type, start_period, end_period):

    # Filter by school type, show results for the provided time period range, show number of authorised absences (sess_authorised)
    result = df.filter((col("geographic_level") == "National") & 
                       (col("school_type") == school_type) &
                       (col("time_period").between(int(start_period), int(end_period)))) \
            .select("time_period", "school_type", "sess_authorised")
    result.show(truncate=False)

    result = df.filter((col("geographic_level") == "National") & 
                       (col("school_type") == school_type) &
                       (col("time_period").between(int(start_period), int(end_period)))) \
            .select("time_period", "school_type", "sess_auth_appointments",
                    "sess_auth_excluded", "sess_auth_ext_holiday", "sess_auth_holiday",
                    "sess_auth_illness", "sess_auth_other", "sess_auth_religious", "sess_auth_study",
                    "sess_auth_traveller")
    result.show(truncate=False)

def search_unauthorised_absences_by_region_type(df, year):

    # Show all unauthorised absences for a provided year, filter by region type and show unauthorised absences
    result = df.filter((col("geographic_level") == "Regional") &
                          (col("time_period") == year) &
                          (col("school_type") == "Total")) \
                .select("time_period", "region_name", "sess_unauthorised")
    result.show(truncate=False)


def compare_local_authorities_in_year(df, local_authority_1, local_authority_2, year):
    # Compare two local authorities in a specific year using enrolements, sess_authorised_percent, sess_overall_percent, sess_unauthorised_percent, school_type=Total, gepgraphical_level=Local authority

    result = df.filter((col("geographic_level") == "Local authority") &
                          (col("time_period") == year) &
                          (col("school_type") == "Total") &
                          (col("la_name").isin(local_authority_1, local_authority_2))) \
                 .select("time_period", "la_name", "enrolments", "sess_authorised_percent",
                            "sess_overall_percent", "sess_unauthorised_percent", "sess_unauth_late", 
                            "sess_auth_appointments", "sess_auth_excluded") 
    result.show(truncate=False)

def performance_of_regions(df):
    # Explore the performance of regions over all time time periods (200607-201819)
    # Questions: 
    # Are there any regions that have improved in pupil attendance over the years?
    # Are there any regions that have worsened?
    # Which is the overall best/worst region for pupil attendance?

    output_dir = "output/Part2b"

    result = df.filter((col("geographic_level") == "Regional") &
                            (col("school_type") == "Total")) \
                .select("time_period", "region_name", "enrolments", "sess_authorised_percent",
                            "sess_overall_percent", "sess_unauthorised_percent", "sess_unauth_late", 
                            "sess_auth_appointments", "sess_auth_excluded", "sess_authorised", "sess_unauthorised", "sess_overall") \
                .orderBy("region_name", "time_period")
    result.show(truncate=False)

    rows = result.collect()
    
    regions = ["East Midlands", "East of England", "Inner London", "North East", "North West", "Outer London", "South East", "South West", "West Midlands", "Yorkshire and the Humber"]
    metrics = ["enrolments", "sess_authorised_percent", "sess_overall_percent", 
               "sess_unauthorised_percent", "sess_unauth_late", "sess_auth_appointments", 
               "sess_auth_excluded", "sess_authorised", "sess_unauthorised", "sess_overall"]
    periods = ["200607", "200708", "200809", "200910", "201011", "201112", "201213", "201314", "201415", "201516", "201617", "201718", "201819"]

    for metric in metrics:
        plt.figure(figsize=(12,6))  # Start a new figure for this metric
        for region in regions:
            region_data = []
            for period in periods:
                # Find the row that matches both the region and the period.
                # If no row is found for a particular period, append None (or you could use np.nan)
                value = None
                for row in rows:
                    if row.region_name == region and int(row.time_period) == int(period):
                        value = row[metric]
                        break
                region_data.append(value)
            
            # Plot the time-series for this region.
            plt.plot(periods, region_data, marker='o', label=region)
        
        plt.title(f"{metric} over Time for All Regions")
        plt.xlabel("Time Period")
        plt.ylabel(metric)
        plt.xticks(rotation=90)  
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.ticklabel_format(style='plain', axis='y')
        plt.tight_layout() 
        plt.savefig(f"{output_dir}/{metric}.png")
        plt.close() 


def analysis_school_type_pupil_absences_and_location_of_school(df):
    """
    Explore the link between school type, pupil absences and location.
    
    This function:
      - Computes the average absence percentage (using 'sess_overall_percent')
        for each school type and region.
      - Creates a grouped bar chart comparing average absence percentages by region.
      - Constructs a boxplot showing the distribution of absence percentages by school type.
      - Performs a one-way ANOVA test to assess if differences between school types are statistically significant.
      
    The analysis is done using Spark DataFrame operations only, without relying on Pandas.
    """
    # Ensure output directory exists for saving charts
    output_dir = "output/Part3"
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data for Regional level and for the three school types of interest
    df_filtered = df.filter((F.col("geographic_level") == "Regional") &
                              (F.col("school_type").isin("Special", "State-funded secondary", "State-funded primary")))
    
    # Compute the average absence percentage, and average absence count per region and school type
    agg_df = df_filtered.groupBy("region_name", "school_type") \
                        .agg(F.avg("sess_overall_percent").alias("avg_absence_percent"), 
                             F.avg("sess_overall").alias("avg_absence_count"))



    
    # Collect the aggregated results (should be small) to build a Python dictionary
    data = agg_df.collect()
    region_data = {}
    regions_set = set()
    for row in data:
        region = row["region_name"]
        stype = row["school_type"]
        avg_value = row["avg_absence_percent"]
        avg_value_count = int(row["avg_absence_count"])
        regions_set.add(region)
        if region not in region_data:
            region_data[region] = {"Special": None, "State-funded secondary": None, "State-funded primary": None}
        region_data[region][stype] = avg_value
        region_data[region][stype + "_count"] = avg_value_count
    regions = sorted(list(regions_set))


    
    # Prepare lists for plotting: one value per region for each school type
    special_vals = [region_data[r]["Special"] for r in regions]
    secondary_vals = [region_data[r]["State-funded secondary"] for r in regions]
    primary_vals = [region_data[r]["State-funded primary"] for r in regions]

    special_vals_count = [region_data[r]["Special_count"] for r in regions]
    secondary_vals_count = [region_data[r]["State-funded secondary_count"] for r in regions]
    primary_vals_count = [region_data[r]["State-funded primary_count"] for r in regions]
    
    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(len(regions)))
    bar_width = 0.25
    
    ax.bar([xi - bar_width for xi in x], special_vals, width=bar_width, label="Special")
    ax.bar(x, secondary_vals, width=bar_width, label="State-funded secondary")
    ax.bar([xi + bar_width for xi in x], primary_vals, width=bar_width, label="State-funded primary")
    
    ax.set_xlabel("Region")
    ax.set_ylabel("Average Absence Percentage")
    ax.set_title("Average Pupil Absence Percentage by School Type and Region")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/grouped_bar_absences.png", bbox_inches='tight')
    plt.close()
    
    

    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(len(regions)))
   
    ax.plot(x, special_vals_count, marker='o', label="Special")
    ax.plot(x, secondary_vals_count, marker='o', label="State-funded secondary")
    ax.plot(x, primary_vals_count, marker='o', label="State-funded primary")
    ax.set_xlabel("Region")
    ax.set_ylabel("Average Absence")
    ax.set_title(f"Average Pupil Absence Count for School Type by Region")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha="right")
    
    plt.ticklabel_format(style='plain', axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_absences.png", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(len(regions)))
   
    ax.plot(x, special_vals_count, marker='o', label="Special")
    ax.set_xlabel("Region")
    ax.set_ylabel("Average Absence")
    ax.set_title(f"Average Pupil Absence Count for School Type Special")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha="right")
    
    plt.ticklabel_format(style='plain', axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_absences_speical.png", bbox_inches='tight')
    plt.close()


def main():

    spark = SparkSession.builder.appName("Absence Data Analysis").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    read_in_dataset(spark)

    df = spark.read.parquet(parquet_file_path)

    while True:

        print("\nPlease select an option:")
        print("\n1 Part 1.a) : Show enrolement for a local authority across time periods")
        print("\n2 Part 1.b) : Show authorised absences for a school type for a specific time period")
        print("\n3 Part 1.c) : Show all unauthorised absences in a region type for a year")
        print("\n4 Part 2.a) : Compare two local authorities in a specific year")
        print("\n5 Part 2.b) : Explore the performance of regions over all time periods")
        print("\n6 Part 3 : Explore link between school type, pupil absences and location of school")
        print("\nType 'exit' to exit the program")
        
        choice = input("\nEnter your query choice (1, 2, 3, 4, 5, 6): ")

        if choice == "1":
            
            local_authority = input("\nEnter the local authority: ")
            search_enrolement_by_local_authority(df, local_authority)

        elif choice == "2":

            school_type = input("\nEnter the school type: ")
            start_period = input("\nEnter the start time period (200607-201819): ")
            end_period = input("\nEnter the end time period (200607-201819): ")

            if start_period > end_period:
                print("\nInvalid time period")
                continue

            search_authorised_absences_by_school_type(df, school_type, start_period, end_period)

        elif choice == "3":

            time_period = input("\nEnter the time period (200607-201819): ")
            search_unauthorised_absences_by_region_type(df, time_period)

        elif choice == "4":
            local_authority_1 = input("\nEnter the first local authority: ")
            local_authority_2 = input("\nEnter the second local authority: ")
            if local_authority_1 == local_authority_2:
                print("\nLocal authorities should be different")
                continue
            year = input("\nEnter the year (200607-201819): ")
            compare_local_authorities_in_year(df, local_authority_1, local_authority_2, year)
        
        elif choice == "5":
            performance_of_regions(df)

        elif choice == "6":
            analysis_school_type_pupil_absences_and_location_of_school(df)
        elif choice == "exit":
            break

    spark.stop()
        


if __name__ == "__main__":
    main()