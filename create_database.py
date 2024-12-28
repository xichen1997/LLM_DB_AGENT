from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import random

# Create database engine
engine = create_engine('sqlite:///sales_database.db')
Base = declarative_base()

class Product(Base):
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    category = Column(String(50))
    price = Column(Float)

class Sale(Base):
    __tablename__ = 'sales'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'))
    quantity = Column(Integer)
    sale_date = Column(DateTime)
    total_amount = Column(Float)
    region = Column(String(50))

def create_sample_data():
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Sample products
    products = [
        Product(name='Laptop', category='Electronics', price=999.99),
        Product(name='Smartphone', category='Electronics', price=699.99),
        Product(name='Headphones', category='Electronics', price=199.99),
        Product(name='T-shirt', category='Clothing', price=29.99),
        Product(name='Jeans', category='Clothing', price=79.99),
        Product(name='Sneakers', category='Footwear', price=89.99),
        Product(name='Coffee Maker', category='Appliances', price=149.99),
        Product(name='Blender', category='Appliances', price=79.99)
    ]
    
    session.add_all(products)
    session.commit()
    
    # Sample regions
    regions = ['North', 'South', 'East', 'West']
    
    # Generate sales data for the last 6 months
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=180)
    
    sales = []
    current_date = start_date
    
    while current_date <= end_date:
        # Generate 1-5 sales per day
        for _ in range(random.randint(1, 5)):
            product = random.choice(products)
            quantity = random.randint(1, 5)
            region = random.choice(regions)
            
            sale = Sale(
                product_id=product.id,
                quantity=quantity,
                sale_date=current_date,
                total_amount=product.price * quantity,
                region=region
            )
            sales.append(sale)
        
        current_date += datetime.timedelta(days=1)
    
    session.add_all(sales)
    session.commit()
    session.close()

if __name__ == "__main__":
    create_sample_data()
    print("Sample database created successfully!") 