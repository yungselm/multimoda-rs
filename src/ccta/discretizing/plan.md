# Step-by-step plan of discretizing library
For a fixed distance on a centerline project points 
1. Algorithm to split Centerline into branches
2. Function to close caps after removing RCA or LCA -> add as additional entry to results
3. Go along every centerline branch project closest point onto plane (fixed distance) translate into PyGeometry object. Use prelabeled points from results
4. Either return all Geometries or create a super object with a geometry for every branch