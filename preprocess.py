import os
import shutil
import pandas as pd
import csv
# Define directories
source_dir = '/media/volume/shape/HemisphereSeparated/Processed/ClusteringAnalysis/FiberClustering/HemisphereSeparated/DWISpace'
destination_base_dir = '/media/volume/shape/TractGeoNet-main/data' 
csv_file = '/media/volume/shape/DSIStudio Shape Measures/DSI_measures_finalized/100206-left.csv'

# Load the CSV file and extract the first column (Cluster IDs)
df = pd.read_csv(csv_file)
selected_clusters = df.iloc[:, 0].str.replace('cluster_', '').astype(str).tolist()  


cluster_annotation_map = {}
with open('/home/exouser/Downloads/FiberClusterAnnotation_k0800_v1.0.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  
    for row in reader:
        cluster_id = row[0].split('_')[1]  
        annotation = row[1]
        cluster_annotation_map[cluster_id] = annotation

subject_ids = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

for subject_id in subject_ids:
    hemisphere_dir = os.path.join(source_dir, subject_id, 'tracts_left_hemisphere')
    
    if os.path.exists(hemisphere_dir):
        cluster_files = [f for f in os.listdir(hemisphere_dir) if f.endswith('.vtp') and f.startswith('cluster_')]
        
        for cluster_file in cluster_files:
            cluster_id = cluster_file.split('_')[1].split('.')[0]
            
            if cluster_id in selected_clusters:
                if cluster_id in cluster_annotation_map:
                    annotation = cluster_annotation_map[cluster_id]
                    
                    destination_dir = os.path.join(destination_base_dir, annotation)
                    
                    if not os.path.exists(destination_dir):
                        os.makedirs(destination_dir)
                    
                    source_file = os.path.join(hemisphere_dir, cluster_file)
                    
                    destination_file = os.path.join(destination_dir, f'{subject_id}_Cluster_{cluster_id}.vtp')
                    
                    shutil.copy(source_file, destination_file)
                    print(f"Copied {source_file} to {destination_file}")
    else:
        print(f"Hemisphere directory for subject {subject_id} does not exist.")


