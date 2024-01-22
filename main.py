import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import SemanticSegmentationDataset
from models.unet import UNet

# define hyperparameters
hyperparameters = {}
hyperparameters['lr'] = 1e-3
hyperparameters['batch_size'] = 2
hyperparameters['epochs'] = 100
hyperparameters['loss'] = 'CE'
# hyperparameters['loss'] = 'Focal'
# hyperparameters['loss'] = 'Dice'
# hyperparameters['loss'] = 'Combo'



# potential categories, choose one set
categories = {}
categories["Class Rarely Or Never Used"] = ["Other Material", "DisturbeView", "VesselLinked", "SolidLargChunk", "PropertiesMaterialInFront", "Other Material"]
categories["Vessel Type Class"] = ["Vessel", "Syringe", "Pippete", "Tube", "IVBag", "DripChamber", "IVBottle", "Beaker", "RoundFlask", "Cylinder", "SeparatoryFunnel", "Funnel", "Burete", "ChromatographyColumn", "Condenser", "Bottle", "Jar", "Connector", "Flask", "Cup", "Bowl", "Erlenmeyer", "Vial", "Dish", "HeatingVessel", "Tube"]
categories["Vessel Properties Class"] = ["Transparent", "SemiTrans", "Opaque", "DisturbeView", "VesselInsideVessel"]
categories['Vessel Part Class'] = ["Cork", "Label", "Part", "Spike", "Valve", "MagneticStirer", "Thermometer", "Spatula", "Holder", "Filter", "PipeTubeStraw"]
categories["Material Property Class"] = ["MaterialOnSurface", "MaterialScattered", "PropertiesMaterialInsideImmersed", "PropertiesMaterialInFront"]
categories["Material Type Class"] =  ["Liquid", "Foam", "Suspension", "Solid", "Powder", "Urine", "Blood", "Gel", "Granular", "SolidLargChunk", "Vapor", "Other Material", "Filled"]
categories_set = "Vessel Part Class"


# data set
train_data = SemanticSegmentationDataset('train', categories=categories[categories_set])
test_data = SemanticSegmentationDataset('test', categories=categories[categories_set])

train_loader = DataLoader(train_data, batch_size=hyperparameters['batch_size'])
test_loader = DataLoader(test_data, batch_size=hyperparameters['batch_size'])


# init model and opt
# out channels == num classes
model = UNet(in_channels=3, out_channels=len(categories[categories_set]))

optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'])
criterion = nn.CrossEntropyLoss()


# one full pass of the shuffled data set with weight updates
def train():
    model.train()
    total_loss = 0.

    for idx, (image, semantic_maps) in enumerate(train_loader):
        # TODO: doublecheck the handling of base and label samples for correct loss calculation
        print(len(categories[categories_set]))
        print(image.shape, semantic_maps.shape)
        assert(0)

        model_output = model(image)

        optimizer.zero_grad()
        loss = criterion(model_output, semantic_maps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss, total_loss / len(train_loader)


# one full pass through the eval set to assess performance
def evaluate():
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for idx, (image, semantic_maps) in enumerate(test_loader):
            # TODO: doublecheck the handling of base and label samples for correct loss calculation
            model_output = model(image)

            loss = criterion(model_output, semantic_maps)
            total_loss += loss.item()
    
    return total_loss, total_loss / len(train_loader)


# main loop
for e in range(hyperparameters['epochs']):
    train_loss, avg_train_loss = train()
    test_loss, avg_test_loss = evaluate()
    print("Epoch %d Training Loss: %.4f. Validation Loss: %.4f. " % (e, avg_train_loss, avg_test_loss))