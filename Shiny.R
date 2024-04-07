library(shiny)
library(randomForest)

# Define UI for application 
ui <- fluidPage(
  
  titlePanel("Random Forest Model Predictor"),
  
  sidebarLayout(
    sidebarPanel(
      # Input fields for variables
      numericInput("size_hector", "Size (Hector):", value = 1),
      selectInput("status_land", "Land Status:", choices = c("mixed", "owner", "share"), selected = "owner"),
      selectInput("varieties", "Varieties:", choices = c("high", "mixed", "trad"), selected = "trad"),
      numericInput("seed_kg", "Seed (kg):", value = 1),
      numericInput("urea_kg", "Urea (kg):", value = 1),
      numericInput("phosphate_kg", "Phosphate (kg):", value = 1),
      numericInput("pesticide_rs", "Pesticide (Rs):", value = 1),
      numericInput("price_seed_rs.kg", "Price of Seed (Rs/kg):", value = 1),
      numericInput("price_urea", "Price of Urea:", value = 1),
      numericInput("price_phosphate", "Price of Phosphate:", value = 1),
      numericInput("total_working_labor_hrs", "Total Working Labor Hours:", value = 1),
      numericInput("wage..hrs", "Wage (per hour):", value = 1),
      selectInput("region", "Region:", choices = c("ciwangi", "gunungwangi", "langan", "malausma", "sukaambit", "wargabinangun"), selected = "wargabinangun"),
      numericInput("net_output", "Net Output:", value = 1),
      numericInput("price.kg", "Price (per kg):", value = 1)
    ),
    
    mainPanel(
      textOutput("prediction")
    )
  )
)

# Define server logic 
server <- function(input, output) {
  
  # Load the model
  load("rf_model.RData")
  
  # Reactive function for model prediction
  output$prediction <- renderText({
    # Prepare input data
    input_data <- data.frame(
      size_hector = input$size_hector,
      status_land = factor(input$status_land, levels = c("mixed", "owner", "share")),
      varieties = factor(input$varieties, levels = c("high", "mixed", "trad")),
      seed_kg = input$seed_kg,
      urea_kg = input$urea_kg,
      phosphate_kg = input$phosphate_kg,
      pesticide_rs = input$pesticide_rs,
      price_seed_rs.kg = input$price_seed_rs.kg,
      price_urea = input$price_urea,
      price_phosphate = input$price_phosphate,
      Total_working_labor_hrs = input$total_working_labor_hrs,
      wage..hrs = input$wage..hrs,
      region = factor(input$region, levels = c("ciwangi", "gunungwangi", "langan", "malausma", "sukaambit", "wargabinangun")),
      net_output = input$net_output,
      price.kg = input$price.kg
    )
    
    # Make prediction using the loaded Random Forest model
    prediction <- predict(rf_model, input_data)  # Use rf_model instead of rf_model
    
    # Output the prediction
    paste("Predicted Gross Output (kg):", prediction)
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
