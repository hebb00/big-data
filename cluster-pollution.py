%pyspark
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.clustering import KMeans
from pyspark.sql import Row
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize


%pyspark
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType, DoubleType, StringType


schema = StructType([
    StructField("sensor_id", IntegerType(), True),
    StructField("sensor_type", StringType(), True),
    StructField("location", IntegerType(), True),
    StructField("lat", DoubleType(), True),
    StructField("lon", DoubleType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("P1", DoubleType(), True),
    StructField("durP1", DoubleType(), True),
    StructField("ratioP1", DoubleType(), True),
    StructField("P2", DoubleType(), True),
    StructField("durP2", DoubleType(), True),
    StructField("ratioP2", DoubleType(), True),
])


def spark_read_pollution(filepath):
    return (
        spark
        .read
        .csv(
            filepath,
            header=True,
            sep=";",
            # mode="DROPMALFORMED",
            timestampFormat="yyyy-MM-dd'T'HH:mm:ss",
            schema=schema,
            ignoreLeadingWhiteSpace=True,
            ignoreTrailingWhiteSpace=True,
        )
    )

pollution = spark_read_pollution("hdfs://bdgtm:8020/pm_data/DE_2024-01_sds011.csv")
%pyspark
pollution = pollution.filter((pollution.P1.isNotNull()) & (pollution.P2.isNotNull()))
pollution.cache()
pollution.count()

%pyspark

# Example: 10 bins for the "value" column
hist_data = pollution.select("p1").rdd.flatMap(lambda x: x).histogram(100)

bins = hist_data[0]      # Bin edges (length = num_bins + 1)
counts = hist_data[1]    # Counts per bin (length = num_bins)
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(counts))]

plt.bar(bin_centers, counts, width=(bins[1] - bins[0]), edgecolor='black')
plt.xlabel("P1")
plt.ylabel("counts")
plt.title("Histogram before removing outliers")
plt.show()
%pyspark
from pyspark.sql import functions as F

%pyspark
from pyspark.sql import functions as F

# Example: compute z-score for column 'value'
stats = pollution.select(
    F.mean("P1").alias("mean_p1"),
    F.stddev("P1").alias("std_p1"),
    F.mean("P2").alias("mean_p2"),
    F.stddev("P2").alias("std_p2"),
    F.mean("lat").alias("mean_lat"),
    F.stddev("lat").alias("std_lat"),
    F.mean("lon").alias("mean_lon"),
    F.stddev("lon").alias("std_lon")
).collect()[0]

mean_p1, std_p1 = stats['mean_p1'], stats['std_p1']
mean_p2, std_p2 = stats['mean_p2'], stats['std_p2']
mean_lat, std_lat = stats['mean_lat'], stats['std_lat']
mean_lon, std_lon = stats['mean_lon'], stats['std_lon']


df_zscore = pollution.withColumn(
    "p1_zscore", (F.col("P1") - F.lit(mean_p1)) / F.lit(std_p1)
).withColumn(
    "p2_zscore", (F.col("P2") - F.lit(mean_p2)) / F.lit(std_p2)
).withColumn(
    "lat_zscore", (F.col("lat") - F.lit(mean_lat)) / F.lit(std_lat)
).withColumn(
    "lon_zscore", (F.col("lon") - F.lit(mean_lon)) / F.lit(std_lon)
)
df_zscore.show(10)


df_zscore = df_zscore.filter(
    (F.col("p1_zscore") >= 2) & 
    (F.col("p1_zscore") < 3) & 
    (F.col("p2_zscore") >= 2) &
    (F.col("p2_zscore") < 3)  
)
df_zscore.count()

%pyspark
hist_data = df_zscore.select("p1_zscore").rdd.flatMap(lambda x: x).histogram(100)

bins = hist_data[0]      # Bin edges (length = num_bins + 1)
counts = hist_data[1]    # Counts per bin (length = num_bins)

bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(counts))]

plt.bar(bin_centers, counts, width=(bins[1] - bins[0]), edgecolor='black')
plt.xlabel("P1")
plt.ylabel("count")
plt.title("Histogram after removing outliers")
plt.show()
%pyspark



from pyspark.ml.feature import VectorAssembler, StandardScaler

# Define your numerical columns
numeric_cols = ['lon_zscore', 'lat_zscore', 'p1_zscore', 'p2_zscore']   # Replace with actual column names  'p1_zscore'

# Assemble features into a single vector column
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
df_vector = assembler.transform(df_zscore)


df_vector.select("features").show(5, truncate=False)
k_values = list(range(2, 11))  

wssse_list = []



for k in k_values:

    kmeans = KMeans(featuresCol='features', predictionCol='cluster', k=k, seed=42)

    model = kmeans.fit(df_vector)

    wssse = model.summary.trainingCost

    wssse_list.append((k, wssse))
    
elbow_data = spark.createDataFrame([Row(k=k, wssse=wssse) for k, wssse in wssse_list])

elbow_data.orderBy("k").show()
%pyspark
ks = [row.k for row in elbow_data.collect()]
wssse = [row.wssse for row in elbow_data.collect()]

plt.plot(ks, wssse, marker='o')
plt.xlabel('k')
plt.ylabel('WSSSE')
plt.title('Elbow Method for Optimal k')
plt.show()

%pyspark

# Train KMeans model on the full scaled dataset with k=4
kmeans = KMeans(featuresCol='features', predictionCol='cluster', k=5, seed=42)
model = kmeans.fit(df_vector)

# Predict clusters
df_clustered = model.transform(df_vector)

# Show some clustered results
df_clustered.select('lon', 'lat',  'cluster').show(10)  #  'p1_zscore'

# Optional: Show cluster centers
centers = model.clusterCenters()
print("Cluster Centers:")
for i, center in enumerate(centers):
    print(f"Cluster {i}: {center}")

%pyspark
import matplotlib.cm as cm

plot_data = df_clustered.select("lat","lon","cluster").limit(200000).collect()
lats = [row['lat'] for row in plot_data]
lons = [row['lon'] for row in plot_data]
clusters = [row['cluster'] for row in plot_data]

plt.figure(figsize=(8,10))

cmap = cm.get_cmap('rainbow', 5)
colors = [cmap(i / 4) for i in range(4)]  

for cluster_id in range(4):
    cluster_lons = [lons[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
    cluster_lats = [lats[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
    plt.scatter(cluster_lons, cluster_lats,
                color=colors[cluster_id],
                label=f'Cluster {cluster_id + 1}',
                alpha=0.6, s=10)

plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("kmeans clustering")
plt.grid(True)
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()