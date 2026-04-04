# Project Methodology

## 1. Approach
This project follows the CRISP-DM methodology:
1. Business Understanding — Define the problem
2. Data Understanding — Explore the dataset
3. Data Preparation — Clean and preprocess
4. Modelling — Build ML models
5. Evaluation — Measure performance
6. Deployment — Flask web application

## 2. Tools Used
- Python 3.11
- Jupyter Notebooks (VS Code)
- Pandas, NumPy — Data processing
- Matplotlib, Seaborn — Visualizations
- Scikit-learn — Machine learning
- Flask — Web application
- AWS S3 — Cloud storage
- GitHub — Version control

## 3. Decisions Made
- Chose E-Commerce Shipping dataset for uniqueness
- Chose Random Forest over Logistic Regression
  due to higher accuracy (~69% vs ~67%)
- Used K=3 clusters based on Elbow Method analysis
- Local-first development approach before cloud upload
- Switched from Azure to AWS due to account eligibility

## 4. Assumptions
- Dataset represents real shipment records accurately
- Late_Delivery=1 means package did NOT reach on time
- Outliers retained as valid business scenarios

## 5. Challenges
- Kernel connection issues in VS Code — fixed by
  reinstalling ipykernel
- Seaborn palette error — fixed by changing dict to list
- Azure account not eligible — switched to AWS free tier
- Cloud trial timing — completed coding locally first

## 6. Cloud Setup
- Platform: AWS (Amazon Web Services)
- Service: AWS S3 (Simple Storage Service)
- Bucket: portfolio-cw2-ecommerce
- Region: Asia Pacific (Mumbai) ap-south-1
- Files uploaded:
  - E Commerce.csv (raw dataset)
  - cleaned_data.csv (processed data)
  - clustered_data.csv (clustered data)