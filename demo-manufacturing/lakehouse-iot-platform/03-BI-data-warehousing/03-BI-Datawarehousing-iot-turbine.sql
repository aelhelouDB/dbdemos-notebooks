-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC
-- MAGIC # Your Lakehouse is the best Warehouse
-- MAGIC
-- MAGIC Traditional Data Warehouses can’t keep up with the variety of data and use cases. Business agility requires reliable, real-time data, with insight from ML models.
-- MAGIC
-- MAGIC Working with the lakehouse unlock traditional BI analysis but also real time applications having a direct connection to your entire data, while remaining fully secured.
-- MAGIC
-- MAGIC <br>
-- MAGIC
-- MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/raw/main/images/dbsql.png" width="700px" style="float: left" />
-- MAGIC
-- MAGIC <div style="float: left; margin-top: 240px; font-size: 23px">
-- MAGIC   Instant, elastic compute<br>
-- MAGIC   Lower TCO with Serveless<br>
-- MAGIC   Zero management<br><br>
-- MAGIC
-- MAGIC   Governance layer - row level<br><br>
-- MAGIC
-- MAGIC   Your data. Your schema (star, data vault…)
-- MAGIC </div>
-- MAGIC
-- MAGIC
-- MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
-- MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=lakehouse&notebook=03-BI-Datawarehousing-iot-turbine&demo_name=lakehouse-patient-readmission&event=VIEW">

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC # BI & Datawarehousing with Databricks SQL
-- MAGIC
-- MAGIC <img style="float: right; margin-top: 10px" width="500px" src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/refs/heads/main/images/manufacturing/lakehouse-iot-turbine/team_flow_alice.png" />
-- MAGIC
-- MAGIC Our datasets are now properly ingested, secured, with a high quality and easily discoverable within our organization.
-- MAGIC
-- MAGIC Let's explore how Databricks SQL support your Data Analyst team with interactive BI and start analyzing our sensor informations.
-- MAGIC
-- MAGIC To start with Databricks SQL, open the SQL view on the top left menu.
-- MAGIC
-- MAGIC You'll be able to:
-- MAGIC
-- MAGIC - Create a SQL Warehouse to run your queries
-- MAGIC - Use DBSQL to build your own dashboards
-- MAGIC - Plug any BI tools (Tableau/PowerBI/..) to run your analysis
-- MAGIC
-- MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
-- MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fmanufacturing%2Flakehouse_iot_turbine%2Fbi&dt=LAKEHOUSE_MANUFACTURING_TURBINE">

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC ## Databricks SQL Warehouses: best-in-class BI engine
-- MAGIC
-- MAGIC <img style="float: right; margin-left: 10px" width="600px" src="https://www.databricks.com/wp-content/uploads/2022/06/how-does-it-work-image-5.svg" />
-- MAGIC
-- MAGIC Databricks SQL is a warehouse engine packed with thousands of optimizations to provide you with the best performance for all your tools, query types and real-world applications. <a href='https://www.databricks.com/blog/2021/11/02/databricks-sets-official-data-warehousing-performance-record.html'>It won the Data Warehousing Performance Record.</a>
-- MAGIC
-- MAGIC This includes the next-generation vectorized query engine Photon, which together with SQL warehouses, provides up to 12x better price/performance than other cloud data warehouses.
-- MAGIC
-- MAGIC **Serverless warehouse** provide instant, elastic SQL compute — decoupled from storage — and will automatically scale to provide unlimited concurrency without disruption, for high concurrency use cases.
-- MAGIC
-- MAGIC Make no compromise. Your best Datawarehouse is a Lakehouse.
-- MAGIC
-- MAGIC ### Creating a SQL Warehouse
-- MAGIC
-- MAGIC SQL Wharehouse are managed by databricks. [Creating a warehouse](/sql/warehouses) is a 1-click step: 

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC
-- MAGIC ## Creating your first Query
-- MAGIC
-- MAGIC <img style="float: right; margin-left: 10px" width="600px" src="https://raw.githubusercontent.com/QuentinAmbard/databricks-demo/main/retail/resources/images/lakehouse-retail/lakehouse-retail-dbsql-query.png" />
-- MAGIC
-- MAGIC Our users can now start running SQL queries using the SQL editor and add new visualizations.
-- MAGIC
-- MAGIC By leveraging auto-completion and the schema browser, we can start running adhoc queries on top of our data.
-- MAGIC
-- MAGIC While this is ideal for Data Analyst to start analysing our customer Churn, other personas can also leverage DBSQL to track our data ingestion pipeline, the data quality, model behavior etc.
-- MAGIC
-- MAGIC Open the [Queries menu](/sql/queries) to start writting your first analysis.

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC
-- MAGIC ## Creating our Wind Turbine farm Dashboards
-- MAGIC
-- MAGIC <img style="float: right; margin-left: 10px" width="600px" src="https://github.com/databricks-demos/dbdemos-resources/raw/main/images/manufacturing/lakehouse-iot-turbine/lakehouse-manuf-iot-dashboard-1.png" />
-- MAGIC
-- MAGIC The next step is now to assemble our queries and their visualization in a comprehensive SQL dashboard that our business will be able to track.
-- MAGIC
-- MAGIC The Dashboard has been loaded for you. Open the <a dbdemos-dashboard-id="turbine-analysis" href="/sql/dashboardsv3/01ef3a4263bc1180931f6ae733179956">Turbine DBSQL Dashboard</a> to start reviewing our our Wind Turbine Farm stats or the <a dbdemos-dashboard-id="turbine-predictive" href="/sql/dashboardsv3/01ef3a4263bc1180931f6ae733179956">DBSQL Predictive maintenance Dashboard</a>.

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC
-- MAGIC ## Using Third party BI tools
-- MAGIC
-- MAGIC <iframe style="float: right" width="560" height="315" src="https://www.youtube.com/embed/EcKqQV0rCnQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
-- MAGIC
-- MAGIC SQL warehouse can also be used with an external BI tool such as Tableau or PowerBI.
-- MAGIC
-- MAGIC This will allow you to run direct queries on top of your table, with a unified security model and Unity Catalog (ex: through SSO). Now analysts can use their favorite tools to discover new business insights on the most complete and freshest data.
-- MAGIC
-- MAGIC To start using your Warehouse with third party BI tool, click on "Partner Connect" on the bottom left and chose your provider.

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC ## Going further with DBSQL & Databricks Warehouse
-- MAGIC
-- MAGIC Databricks SQL offers much more and provides a full warehouse capabilities
-- MAGIC
-- MAGIC <img style="float: right" width="400px" src="https://raw.githubusercontent.com/QuentinAmbard/databricks-demo/main/retail/resources/images/lakehouse-retail/lakehouse-retail-dbsql-pk-fk.png" />
-- MAGIC
-- MAGIC ### Data modeling
-- MAGIC
-- MAGIC Comprehensive data modeling. Save your data based on your requirements: Data vault, Star schema, Inmon...
-- MAGIC
-- MAGIC Databricks let you create your PK/FK, identity columns (auto-increment): `dbdemos.install('identity-pk-fk')`
-- MAGIC
-- MAGIC ### Data ingestion made easy with DBSQL & DBT
-- MAGIC
-- MAGIC Turnkey capabilities allow analysts and analytic engineers to easily ingest data from anything like cloud storage to enterprise applications such as Salesforce, Google Analytics, or Marketo using Fivetran. It’s just one click away. 
-- MAGIC
-- MAGIC Then, simply manage dependencies and transform data in-place with built-in ETL capabilities on the Data Intelligence Platform (Delta Live Table), or using your favorite tools like dbt on Databricks SQL for best-in-class performance.
-- MAGIC
-- MAGIC ### Query federation
-- MAGIC
-- MAGIC Need to access cross-system data? Databricks SQL query federation let you define datasources outside of databricks (ex: PostgreSQL)
-- MAGIC
-- MAGIC ### Materialized view
-- MAGIC
-- MAGIC Avoid expensive queries and materialize your tables. The engine will recompute only what's required when your data get updated. 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC
-- MAGIC # Taking our analysis one step further: Predicting Churn
-- MAGIC
-- MAGIC Being able to run analysis on our past data already gives us a lot of insight. We can better understand which customers are churning evaluate the churn impact.
-- MAGIC
-- MAGIC However, knowing that we have churn isn't enough. We now need to take it to the next level and build a predictive model to determine our customers at risk of churn to be able to increase our revenue.
-- MAGIC
-- MAGIC Let's see how this can be done with [Databricks Machine Learning notebook]($../04-Data-Science-ML/04.1-automl-iot-turbine-predictive-maintenance) | go [Go back to the introduction]($../00-IOT-wind-turbine-introduction-DI-platform)
