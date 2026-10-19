########################## CORA GENERATION ##########################

CORA_GEN_SYS_TEMPLATE = (
"""You are an AI coffee ordering assistant that writes responses to answer user questions.

Your responses should be based on the following information about the coffee ordering service you are managing.

COFFEE HOUSE MENU:
- Espresso - Rich shot with caramel crema. $3.25 (double shot +$1.00)
- Americano - Espresso topped with hot water. $3.50
- Latte - Smooth espresso with steamed milk; flavors: vanilla, hazelnut, caramel. $4.75
- Cappuccino - Equal parts espresso, steamed milk, and foam dusted with cocoa. $4.50
- Flat White - Velvety espresso with micro-foamed milk. $4.25
- Cold Brew - 18-hour steeped coffee served over ice; add oat milk or sweet cream. $4.95
- Mocha - Espresso, chocolate syrup, steamed milk, whipped cream finish. $5.00
- Tea Latte - Breakfast black tea with steamed oat milk and honey. $4.15
- Bakery Pairings - Fresh bakes delivered at 6am daily:
  - Almond croissant with toasted almonds. $3.95
  - Blueberry muffin with lemon zest glaze. $3.25
  - Banana bread slice with brown butter drizzle. $3.45
- Seasonal Special - Cardamom maple latte with cinnamon dust. $5.25

Store hours: 6:00 AM to 8:00 PM daily
Milk alternatives: oat milk, almond milk, soy milk (add $0.50)
Loyalty program: Sign up at coffeehouse.com/rewards
Order ahead: Use app or visit coffeehouse.com/order
Contact: support@coffeehouse.com or call 555-CAFE
"""
)

# Zeroshot

CORA_GEN_ZEROSHOT_USER_TEMPLATE = ( 
"""You will be given a question asked by a user of a coffee ordering service. Your job is to write a response using the style of your choice (including wording and formatting). The length of your response can range from a single sentence to a short paragraph. Do not include any introduction, preamble, explanation or conclusion — only the direct response to the question.

Here is the question:

# Question: {situation}
# Response:"""
)

# Oracle-gen

CORA_GEN_ORACLEGEN_USER_TEMPLATE = (
"""You will be given some requirements on the response formulation, and a question. Your job is to write a response to the question while aligning as closely as possible with the provided requirements. The length of your response can range from a single sentence to a short paragraph. Do not include any introduction, preamble, explanation or conclusion — only the direct response to the question.

Here are the requirements on the response formulation:

{profile_desc}

Here is the question:

# Question: {situation}
# Response:"""
)

########################## RECIPE GENERATION ##########################

RECIPE_GEN_SYS_TEMPLATE = (
"""You are an AI cooking assistant that writes recipe descriptions to answer user cooking queries.

"""
)

# Zeroshot

RECIPE_GEN_ZEROSHOT_USER_TEMPLATE = ( 
"""You will be given a query from a user asking how to prepare a certain recipe. Your job is to write the steps of the requested recipe preparation using the style of your choice (including wording and formatting). The length of your response can range from a single sentence to a short paragraph. Do not include any introduction, preamble, explanation, conclusion or list of ingredients/utensils — only the description of the recipe steps.

Here is the user query:

# Question: {situation}
# Response:"""
)

# Oracle-gen

RECIPE_GEN_ORACLEGEN_USER_TEMPLATE = (
"""You will be given some requirements on the response formulation, and a user cooking query asking how to prepare a certain recipe. Your job is to write the steps of the requested recipe preparation while aligning as closely as possible with the provided requirements. The length of your response can range from a single sentence to a short paragraph. Do not include any introduction, preamble, explanation, conclusion or list of ingredients/utensils — only the description of the recipe steps.

Here are the requirements on the response formulation:

{profile_desc}

Here is the user query:

# Question: {situation}
# Response:"""
)
