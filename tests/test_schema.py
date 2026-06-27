from src.api.main import CustomerData


def test_customer_schema_valid():
    customer = CustomerData(
        gender="Male",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=12,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="DSL",
        OnlineSecurity="No",
        OnlineBackup="Yes",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=70.5,
        TotalCharges=850.0,
    )

    assert customer.gender == "Male"
    assert customer.tenure == 12
    assert customer.PaymentMethod == "Electronic check"