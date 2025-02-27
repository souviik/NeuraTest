Feature: Product Search
  Scenario: Search for iPhone 16
    Given I navigate to "https://www.amazon.in"
    When I search for "iphone 16"
    Then I should see product results

  Scenario: Validate Price Display
    Given I navigate to "https://www.amazon.in"
    When I search for "iphone 16"
    Then the price should be displayed in INR
    Then print the product and price name in console