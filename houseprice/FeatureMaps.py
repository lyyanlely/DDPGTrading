featuremap={
'MSSubClass': 
       {
        20: 20,
        30: 30,
        40: 40,
        45: 45,
        50: 50,
        60: 60,
        70: 70,
        75: 75,
        80: 80,
        85: 85,
        90: 90,
       120: 120,
       150: 150,
       160: 160,
       180: 180,
       190: 190
       }, 

'MSZoning': 
	{	
       'A': 0,
       'C (all)': 1,
       'FV': 2,
       'I': 3,
       'RH': 4,
       'RL': 5,
       'RP': 6,
       'RM': 7
       }, 
	
'LotFrontage':
       {},

'LotArea':
       {},

'Street':
       {
       'Grvl': 0,
       'Pave': 1
       },
       	
'Alley':
       {
       'Grvl': 1,
       'Pave': 2,
       # 'NA': 0
       },
		
'LotShape':
       {
       'Reg':	0,	
       'IR1': 1,
       'IR2': 2,
       'IR3': 3
       },
       
'LandContour': 
       {
       'Lvl': 0,
       'Bnk': 1,
       'HLS':	2,
       'Low':	3
       },
		
'Utilities': 
	{
       'AllPub': 0,
       'NoSewr': 1,
       'NoSeWa': 2,
       'ELO': 3
       },	
	
'LotConfig':
       {
       'Inside': 0,
       'Corner': 1,
       'CulDSac': 2,
       'FR2': 3,
       'FR3': 4
       },
	
'LandSlope':
	{
       'Gtl': 0,
       'Mod': 1,	
       'Sev': 2
       },
	
'Neighborhood':
       {
       'Blmngtn': 0,
       'Blueste': 1,
       'BrDale': 2,
       'BrkSide': 3,
       'ClearCr': 4, # Clear Creek
       'CollgCr': 5, # College Creek
       'Crawfor': 6, # Crawford
       'Edwards': 7, # Edwards
       'Gilbert': 8, # Gilbert
       'IDOTRR': 9, # Iowa DOT and Rail Road
       'MeadowV': 10, # Meadow Village
       'Mitchel': 11, # Mitchell
       'NAmes': 12, # North Ames
       'NoRidge': 13, # Northridge
       'NPkVill': 14, # Northpark Villa
       'NridgHt': 15, # Northridge Heights
       'NWAmes': 16, # Northwest Ames
       'OldTown': 17, # Old Town
       'SWISU': 18, # South & West of Iowa State University
       'Sawyer': 19, # Sawyer
       'SawyerW': 20, # Sawyer West
       'Somerst': 21, # Somerset
       'StoneBr': 22, # Stone Brook
       'Timber': 23, # Timberland
       'Veenker': 24 # Veenker
       },
			
'Condition1': # Proximity to various conditions
	{
       'Artery': 0, #	Adjacent to arterial street
       'Feedr': 1, #	Adjacent to feeder street	
       'Norm': 4, #	Normal	
       'RRNn': 3, #	Within 200' of North-South Railroad
       'RRAn': 2, #	Adjacent to North-South Railroad
       'PosN': 5, #	Near positive off-site feature--park, greenbelt, etc.
       'PosA': 6, #	Adjacent to postive off-site feature
       'RRNe': 3, #	Within 200' of East-West Railroad
       'RRAe': 2 #	Adjacent to East-West Railroad
       },
	
'Condition2': # Proximity to various conditions (if more than one is present)
	{	
       'Artery': 0, #	Adjacent to arterial street
       'Feedr': 1, #	Adjacent to feeder street	
       'Norm': 4, #	Normal	
       'RRNn': 3, #	Within 200' of North-South Railroad
       'RRAn': 2, #	Adjacent to North-South Railroad
       'PosN': 5, #	Near positive off-site feature--park, greenbelt, etc.
       'PosA': 6, #	Adjacent to postive off-site feature
       'RRNe': 3, #	Within 200' of East-West Railroad
       'RRAe': 2 #	Adjacent to East-West Railroad
       },
	
'BldgType': # Type of dwelling
	{	
       '1Fam': 0, #	Single-family Detached	
       '2fmCon': 1, #	Two-family Conversion; originally built as one-family dwelling
       'Duplex': 4, #	Duplex
       'TwnhsE': 3, #	Townhouse End Unit
       'Twnhs': 2 #	Townhouse Inside Unit
       },
	
'HouseStyle': # Style of dwelling
	{
       '1Story': 0, #	One story
       '1.5Fin': 1, #	One and one-half story: 2nd level finished
       '1.5Unf': .5, #	One and one-half story: 2nd level unfinished
       '2Story': 2, #	Two story
       '2.5Fin': 3, #	Two and one-half story: 2nd level finished
       '2.5Unf': 2.5, #	Two and one-half story: 2nd level unfinished
       'SFoyer': .5, #	Split Foyer
       'SLvl': .5 #	Split Level
       },
	
'OverallQual': # Rates the overall material and finish of the house
       {
       10: 10, #	Very Excellent
       9: 9, #	Excellent
       8: 8, #	Very Good
       7: 7, #	Good
       6: 6, #	Above Average
       5: 5, #	Average
       4: 4, #	Below Average
       3: 3, #	Fair
       2: 2, #	Poor
       1: 1 #	Very Poor
       },
	
'OverallCond': # Rates the overall condition of the house
       {
       10: 10, #	Very Excellent
       9: 9, #	Excellent
       8: 8, #	Very Good
       7: 7, #	Good
       6: 6, #	Above Average	
       5: 5, #	Average
       4: 4, #	Below Average	
       3: 3, #	Fair
       2: 2, #	Poor
       1: 1 #	Very Poor
       },
		
'YearBuilt': # Original construction date
       {},

'YearRemodAdd': # Remodel date (same as construction date if no remodeling or additions)
       {},

'RoofStyle': # Type of roof
       {
       'Flat': 0, #	Flat
       'Gable': 1, #	Gable
       'Gambrel': 2, #	Gabrel (Barn)
       'Hip': 3, #	Hip
       'Mansard': 4, #	Mansard
       'Shed': 5 #	Shed
       },
		
'RoofMatl': # Roof material
       {
       'ClyTile': 0, #	Clay or Tile
       'CompShg': 1, #	Standard (Composite) Shingle
       'Membran': 2, #	Membrane
       'Metal': 3, #	Metal
       'Roll': 4, #	Roll
       'Tar&Grv': 5, #	Gravel & Tar
       'WdShake': 6, #	Wood Shakes
       'WdShngl': 7 #	Wood Shingles
       },
		
'Exterior1st': # Exterior covering on house
       {
       'AsbShng': 0, #	Asbestos Shingles
       'AsphShn': 1, #	Asphalt Shingles
       'BrkComm': 2, #	Brick Common
       'BrkFace': 3, #	Brick Face
       'CBlock': 4, #	Cinder Block
       'CemntBd': 5, #	Cement Board
       'HdBoard': 6, #	Hard Board
       'ImStucc': 7, #	Imitation Stucco
       'MetalSd': 8, #	Metal Siding
       'Other': 9, #	Other
       'Plywood': 10, #	Plywood
       'PreCast': 11, #	PreCast	
       'Stone': 12, #	Stone
       'Stucco': 13, #	Stucco
       'VinylSd': 14, #	Vinyl Siding
       'Wd Sdng': 15, #	Wood Siding
       'WdShing': 16 #	Wood Shingles
       },
	
'Exterior2nd': # Exterior covering on house (if more than one material)
       {
       'AsbShng': 0, #       Asbestos Shingles
       'AsphShn': 1, #       Asphalt Shingles
       'Brk Cmn': 2, #       Brick Common
       'BrkFace': 3, #       Brick Face
       'CBlock': 4, # Cinder Block
       'CmentBd': 5, #       Cement Board
       'HdBoard': 6, #       Hard Board
       'ImStucc': 7, #       Imitation Stucco
       'MetalSd': 8, #       Metal Siding
       'Other': 9, #  Other
       'Plywood': 10, #      Plywood
       'PreCast': 11, #      PreCast       
       'Stone': 12, # Stone
       'Stucco': 13, #       Stucco
       'VinylSd': 14, #      Vinyl Siding
       'Wd Sdng': 15, #      Wood Siding
       'Wd Shng': 16 #      Wood Shingles
       },
	
'MasVnrType': # Masonry veneer type
       {
       'BrkCmn': 0, #	Brick Common
       'BrkFace': 1, #	Brick Face
       'CBlock': 2, #	Cinder Block
       'None': 3, #	None
       'Stone': 4 #	Stone
       },
	
'MasVnrArea': # Masonry veneer area in square feet
       {},

'ExterQual': # Evaluates the quality of the material on the exterior 
	{
       'Ex': 5, #	Excellent
       'Gd': 4, #	Good
       'TA': 3, #	Average/Typical
       'Fa': 2, #	Fair
       'Po': 1 #	Poor
       },
		
'ExterCond': # Evaluates the present condition of the material on the exterior
       {
       'Ex': 5, #     Excellent
       'Gd': 4, #     Good
       'TA': 3, #     Average/Typical
       'Fa': 2, #     Fair
       'Po': 1 #     Poor
       },
		
'Foundation': # Type of foundation
	{	
       'BrkTil': 0, #	Brick & Tile
       'CBlock': 1, #	Cinder Block
       'PConc': 2, #	Poured Contrete	
       'Slab': 3, #	Slab
       'Stone': 4, #	Stone
       'Wood': 5 #	Wood
       },
		
'BsmtQual': # Evaluates the height of the basement
       {
       'Ex': 5, #	Excellent (100+ inches)	
       'Gd': 4, #	Good (90-99 inches)
       'TA': 3, #	Typical (80-89 inches)
       'Fa': 2, #	Fair (70-79 inches)
       'Po': 1, #	Poor (<70 inches
       'NA': 0 #	No Basement
       },
		
'BsmtCond': # Evaluates the general condition of the basement
       {
       'Ex': 5, #	Excellent
       'Gd': 4, #	Good
       'TA': 3, #	Typical - slight dampness allowed
       'Fa': 2, #	Fair - dampness or some cracking or settling
       'Po': 1, #	Poor - Severe cracking, settling, or wetness
       'NA': 0 #	No Basement
       },
	
'BsmtExposure': # Refers to walkout or garden level walls
       {
       'Gd': 4, #	Good Exposure
       'Av': 3, #	Average Exposure (split levels or foyers typically score average or above)	
       'Mn': 2, #	Mimimum Exposure
       'No': 1, #	No Exposure
       'NA': 0 #	No Basement
       },
	
'BsmtFinType1': # Rating of basement finished area
       {
       'GLQ': 6, #	Good Living Quarters
       'ALQ': 5, #	Average Living Quarters
       'BLQ': 4, #	Below Average Living Quarters	
       'Rec': 3, #	Average Rec Room
       'LwQ': 2, #	Low Quality
       'Unf': 1, #	Unfinshed
       'NA': 0 #	No Basement
       },
		
'BsmtFinSF1': # Type 1 finished square feet
       {},

'BsmtFinType2': # Rating of basement finished area (if multiple types)
       {
       'GLQ': 6, #    Good Living Quarters
       'ALQ': 5, #    Average Living Quarters
       'BLQ': 4, #    Below Average Living Quarters      
       'Rec': 3, #    Average Rec Room
       'LwQ': 2, #    Low Quality
       'Unf': 1, #    Unfinshed
       'NA': 0 #     No Basement
       },

'BsmtFinSF2': # Type 2 finished square feet
       {},

'BsmtUnfSF': # Unfinished square feet of basement area
       {},

'TotalBsmtSF': # Total square feet of basement area
       {},

'Heating': # Type of heating
	{	
       'Floor': 1, #	Floor Furnace
       'GasA': 2, #	Gas forced warm air furnace
       'GasW': 3, #	Gas hot water or steam heat
       'Grav': 4, #	Gravity furnace	
       'OthW': 5, #	Hot water or steam heat other than gas
       'Wall': 6 #	Wall furnace
       },
		
'HeatingQC': # Heating quality and condition
       {
       'Ex': 5, #	Excellent
       'Gd': 4, #	Good
       'TA': 3, #	Average/Typical
       'Fa': 2, #	Fair
       'Po': 1 #	Poor
       },
		
'CentralAir': # Central air conditioning
       {
       'N': 0, #	No
       'Y': 1 #	Yes
       },
		
'Electrical': # Electrical system
       {
       'SBrkr': 0, #	Standard Circuit Breakers & Romex
       'FuseA': 1, #	Fuse Box over 60 AMP and all Romex wiring (Average)	
       'FuseF': 2, #	60 AMP Fuse Box and mostly Romex wiring (Fair)
       'FuseP': 3, #	60 AMP Fuse Box and mostly knob & tube wiring (poor)
       'Mix': 4 #	Mixed
       },
		
'1stFlrSF': # First Floor square feet
       {},
 
'2ndFlrSF': # Second floor square feet
       {},

'LowQualFinSF': # Low quality finished square feet (all floors)
       {},

'GrLivArea': # Above grade (ground) living area square feet
       {},

'BsmtFullBath': # Basement full bathrooms
       {},

'BsmtHalfBath': # Basement half bathrooms
       {},

'FullBath': # Full bathrooms above grade
       {},

'HalfBath': # Half baths above grade
       {},

'BedroomAbvGr': # Bedrooms above grade (does NOT include basement bedrooms)
       {},

'KitchenAbvGr': # Kitchens above grade
       {},

'KitchenQual': # Kitchen quality
       {
       'Ex': 5, #	Excellent
       'Gd': 4, #	Good
       'TA': 3, #	Typical/Average
       'Fa': 2, #	Fair
       'Po': 1 #	Poor
       },
       	
'TotRmsAbvGrd': # Total rooms above grade (does not include bathrooms)
       {},

'Functional': # Home functionality (Assume typical unless deductions are warranted)
       {
       'Typ': 0, #	Typical Functionality
       'Min1': 1, #	Minor Deductions 1
       'Min2': 2, #	Minor Deductions 2
       'Mod': 3, #	Moderate Deductions
       'Maj1': 4, #	Major Deductions 1
       'Maj2': 5, #	Major Deductions 2
       'Sev': 6, #	Severely Damaged
       'Sal': 7 #	Salvage only
       },
		
'Fireplaces': # Number of fireplaces
       {
       },

'FireplaceQu': # Fireplace quality
       {
       'Ex': 5, #	Excellent - Exceptional Masonry Fireplace
       'Gd': 4, #	Good - Masonry Fireplace in main level
       'TA': 3, #	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       'Fa': 2, #	Fair - Prefabricated Fireplace in basement
       'Po': 1, #	Poor - Ben Franklin Stove
       'NA': 0 #	No Fireplace
       },
		
'GarageType': # Garage location
	{	
       '2Types': 6, #	More than one type of garage
       'Attchd': 5, #	Attached to home
       'Basment': 4, #	Basement Garage
       'BuiltIn': 3, #	Built-In (Garage part of house - typically has room above garage)
       'CarPort': 2, #	Car Port
       'Detchd': 1, #	Detached from home
       'NA': 0 #	No Garage
       },
		
'GarageYrBlt': # Year garage was build
       {},
		
'GarageFinish': # Interior finish of the garage
       {
       'Fin': 3, #	Finished
       'RFn': 2, #	Rough Finished	
       'Unf': 1, #	Unfinished
       'NA': 0 #	No Garage
       },
		
'GarageCars': # Size of garage in car capacity
       {},

'GarageArea': # Size of garage in square feet
       {},

'GarageQual': # Garage quality
       {
       'Ex': 5, #	Excellent
       'Gd': 4, #	Good
       'TA': 3, #	Typical/Average
       'Fa': 2, #	Fair
       'Po': 1, #	Poor
       'NA': 0 #	No Garage
       },
		
'GarageCond': # Garage condition
       {
       'Ex': 5, #     Excellent
       'Gd': 4, #     Good
       'TA': 3, #     Typical/Average
       'Fa': 2, #     Fair
       'Po': 1, #     Poor
       'NA': 0 #     No Garage
       },
		
'PavedDrive': # Paved driveway
       {
       'Y': 1, #	Paved 
       'P': .5, #	Partial Pavement
       'N': 0 #	Dirt/Gravel
       },
		
'WoodDeckSF': # Wood deck area in square feet
       {},

'OpenPorchSF': # Open porch area in square feet
       {},

'EnclosedPorch': # Enclosed porch area in square feet
       {},

'3SsnPorch': # Three season porch area in square feet
       {},

'ScreenPorch': # Screen porch area in square feet
       {},

'PoolArea': # Pool area in square feet
       {},

'PoolQC': # Pool quality
       {
       'Ex': 5, #     Excellent
       'Gd': 4, #     Good
       'TA': 3, #     Typical/Average
       'Fa': 2, #     Fair
       'Po': 1, #     Poor
       'NA': 0 #     No Pool
       },
		
'Fence': # Fence quality
	{       
       'GdPrv': 4, #       Good Privacy
       'MnPrv': 3, #       Minimum Privacy
       'GdWo': 2, #       Good Wood
       'MnWw': 1, #       Minimum Wood/Wire
       'NA': 0 #       No Fence
       },

'MiscFeature': # Miscellaneous feature not covered in other categories
	{
       'Elev': 5, #       Elevator
       'Gar2': 4, #       2nd Garage (if not described in garage section)
       'Othr': 1, #       Other
       'Shed': 3, #       Shed (over 100 SF)
       'TenC': 2, #       Tennis Court
       'NA': 0 #       None
       },

'MiscVal': # $Value of miscellaneous feature
       {},

'MoSold': # Month Sold (MM)
       {},

'YrSold': # Year Sold (YYYY)
       {},

'SaleType': # Type of sale
	{
       'WD': 0, #        Warranty Deed - Conventional
       'CWD': 1, #       Warranty Deed - Cash
       'VWD': 2, #       Warranty Deed - VA Loan
       'New': 3, #       Home just constructed and sold
       'COD': 4, #       Court Officer Deed/Estate
       'Con': 5, #       Contract 15% Down payment regular terms
       'ConLw': 6, #       Contract Low Down payment and low interest
       'ConLI': 7, #       Contract Low Interest
       'ConLD': 8, #       Contract Low Down
       'Oth': 9 #       Other
       },

'SaleCondition': # Condition of sale
       {
       'Normal': 0, #	Normal Sale
       'Abnorml': 1, #	Abnormal Sale -  trade, foreclosure, short sale
       'AdjLand': 2, #	Adjoining Land Purchase
       'Alloca': 3, #	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
       'Family': 4, #	Sale between family members
       'Partial': 5 #	Home was not completed when last assessed (associated with New Homes)
       },
}
