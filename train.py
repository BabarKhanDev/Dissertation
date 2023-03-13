def train_GAN(
    generator, 
    discriminator,
    optim_g,
    optim_d,
    save_dir):

    ###################################
    # Set up our training environment #
    ###################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, discriminator = generator.to(device), discriminator.to(device)

    patch = (1, 4, 4)

    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss() 

    train_losses = []
    val_losses = []
    best_val_loss = 1000000

    # Loss weight of L1 pixel-wise loss between translated image and real image
    # This value is copied from the PIX2PIX architecture
    lambda_pixel = 100
    num_epochs = 100
    
    #######################
    # Start Training Loop #
    #######################

    print(f'Starting Training for {save_dir}')
    for epoch in range(num_epochs):

        # Go into training mode
        discriminator.train()
        generator.train()

        # Train the model and evaluate on the training set
        total_train_loss = 0
        total_val_loss = 0

        for i, (images, real_labels) in enumerate(train_loader):

            # Adversarial ground truths
            valid = torch.ones((images.size(0), *patch)).to(device)
            fake = torch.zeros((images.size(0), *patch)).to(device)

            # Move images to device and create an image prediction
            images, real_labels = images.to(device), real_labels.to(device)

            ###################
            # Train Generator #
            ###################

            optim_G.zero_grad()

            # GAN Loss

            fake_labels = generator(images)
            pred_fake = discriminator(fake_labels, images)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(fake_labels, real_labels)

            # Logging
            batch_loss = loss_pixel.item() * batch_size
            total_train_loss += batch_loss

            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()
            optim_G.step()

            #######################
            # Train Discriminator #
            #######################

            optim_D.zero_grad()

            # Real Loss
            pred_real = discriminator(real_labels, images)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake Loss
            pred_fake = discriminator(fake_labels.detach(), images)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total Loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optim_D.step()

            # Clear memory
            del images, real_labels, fake_labels 
            torch.cuda.empty_cache() 

        generator.eval()
        with torch.no_grad():
          for i, (images, real_labels) in enumerate(val_loader):

              # Adversarial ground truths
              valid = torch.ones((images.size(0), *patch)).to(device)
              fake = torch.zeros((images.size(0), *patch)).to(device)

              # Move images to device and create an image prediction
              images, real_labels = images.to(device), real_labels.to(device)

              #########################
              # Test Generator On Val #
              #########################

              # GAN Loss

              fake_labels = generator(images)
              loss_pixel = criterion_pixelwise(fake_labels, real_labels)
              batch_loss = loss_pixel.item() * batch_size
              total_val_loss += batch_loss

              # Clear memory
              del images, real_labels, fake_labels 
              torch.cuda.empty_cache() 

        train_losses.append( total_train_loss/ len(train_loader))
        val_losses.append( total_val_loss / len(val_loader))

        print(f'Epoch [{epoch + 1}/{num_epochs}], D: {loss_D.item():.4f}, G: {loss_G.item():.4f},  Val Loss: {val_losses[-1]} Train Loss: {train_losses[-1]}')

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save(discriminator.state_dict(), f'{save_dir}/weights-NETD-best.pkl')
            torch.save(generator.state_dict(), f'{save_dir}/weights-NETG-best.pkl')

    # Use pickle to save the list of train/val losses
    # This will allow us to analyse them later
    save_list_to_file(val_losses  , f'{save_dir}/val_losses')
    save_list_to_file(train_losses, f'{save_dir}/train_losses')

    return best_val_loss

def save_list_to_file(list_to_save, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(list_to_save, fp)

def read_list_from_file(filename):
    with open(filename, 'rb') as fp:
        list_read = pickle.load(fp)
        return list_read