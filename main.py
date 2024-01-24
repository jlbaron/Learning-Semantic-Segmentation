import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import SemanticSegmentationDataset
from models.unet import UNet

# define hyperparameters
hyperparameters = {}
hyperparameters['model_name'] = 'test'
hyperparameters['lr'] = 1e-3
hyperparameters['batch_size'] = 5
hyperparameters['epochs'] = 10
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

# device management, this task should ideally be done on gpu
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

# dice loss option
def dice_loss(input, target, epsilon=1e-6):
    '''
    Dice coefficient is 2*intersection / union+intersection
        where intersection is input*target
        where union is target*target
    
    Dice loss is 1 - dice coefficient
    '''
    input =  torch.flatten(input)
    target = torch.flatten(target)

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)

    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    dice_coefficient = 2 * (intersect / denominator.clamp(min=epsilon))

    return 1. - torch.mean(dice_coefficient)

# combination of dice and ce loss
def combo_loss(input, target):
    return nn.CrossEntropyLoss()(input, target)*0.5 + dice_loss(input, target)*0.5

# data set
train_data = SemanticSegmentationDataset('Train', categories=categories[categories_set])
test_data = SemanticSegmentationDataset('Test', categories=categories[categories_set])

train_loader = DataLoader(train_data, batch_size=hyperparameters['batch_size'], shuffle=True)#, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=hyperparameters['batch_size'])#, num_workers=2, pin_memory=True)


# init model and opt
# out channels == num classes
model = UNet(in_channels=3, out_channels=len(categories[categories_set]), device=device)

optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'])

# TODO: obtain class weights for focal loss
if hyperparameters['loss'] == 'CE':
    criterion = nn.CrossEntropyLoss()
elif hyperparameters['loss'] == 'Dice':
    criterion = dice_loss
elif hyperparameters['loss'] == 'Combo':
    criterion = combo_loss



# one full pass of the shuffled data set with weight updates
def train():
    model.train()
    total_loss = 0.

    for idx, (image, semantic_maps) in enumerate(train_loader):
        model_output = model(image.to(device))
        semantic_maps = semantic_maps.to(device)

        optimizer.zero_grad()
        loss = criterion(model_output, semantic_maps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        print(f"Train iter: {idx},  Loss: {total_loss}")
    
    return total_loss, total_loss / len(train_loader)


# one full pass through the eval set to assess performance
def evaluate():
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for idx, (image, semantic_maps) in enumerate(test_loader):
            model_output = model(image.to(device))

            loss = criterion(model_output, semantic_maps.to(device))
            total_loss += loss.item()
    
    return total_loss, total_loss / len(train_loader)



# main loop
train_losses, test_losses = [], []
for e in range(hyperparameters['epochs']):
    train_loss, avg_train_loss = train()
    test_loss, avg_test_loss = evaluate()
    print("Epoch %d Training Loss: %.4f. Validation Loss: %.4f. " % (e, avg_train_loss, avg_test_loss))
    # Save the trained model
    model_path = os.path.join('checkpoints', f"{hyperparameters['model_name']}_trained.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved to '{model_path}'.")
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)


# Plots
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Training and Testing Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("visualizations\\training_curves\\loss.png")
plt.show()
# plt.plot(train_accuracies, label="Training Accuracies")
# plt.plot(val_accuracies, label="Validation Accuracies")
# plt.title("Training and Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.savefig("figures\\acc.png")
# plt.show()

# TODO: implement per pixel accuracy