"""python -m unittest quad_torch.test -v"""

import unittest
import torch

from quad_torch import QUAD, Procrustes


class Test(unittest.TestCase):
    def test_import(self):
        print("\n=== Testing imports ===")
        self.assertTrue(callable(QUAD))
        print("✓ QUAD imported successfully")
        self.assertTrue(callable(Procrustes))
        print("✓ Procrustes imported successfully")

    def _test_step_updates_params_helper(self, optimizer_class):
        print(f"\n=== Testing parameter updates for {optimizer_class.__name__} ===")
        torch.manual_seed(0)

        model = torch.nn.Sequential(
            torch.nn.Linear(8, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 2),
        )

        initial_params = [p.detach().clone() for p in model.parameters()]

        optimizer = optimizer_class(model.parameters())

        x = torch.randn(16, 8)
        target = torch.zeros(16, 2)
        criterion = torch.nn.MSELoss()

        model.train()
        
        with torch.no_grad():
            initial_loss = criterion(model(x), target).item()
        print(f"Initial loss: {initial_loss:.6f}")
        
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), target)
        loss.backward()
        
        grad_norms = []
        for i, p in enumerate(model.parameters()):
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                grad_norms.append(grad_norm)
                print(f"Param {i} gradient norm: {grad_norm:.6f}")
        
        optimizer.step()

        with torch.no_grad():
            final_loss = criterion(model(x), target).item()
        print(f"Final loss: {final_loss:.6f}")
        print(f"Loss change: {final_loss - initial_loss:.6f}")

        changed = False
        total_param_change = 0.0
        for i, (param, param0) in enumerate(zip(model.parameters(), initial_params)):
            self.assertTrue(torch.isfinite(param).all())
            param_change = (param - param0).norm().item()
            total_param_change += param_change
            print(f"Param {i} change norm: {param_change:.6f}")
            if not torch.allclose(param, param0):
                changed = True

        print(f"Total parameter change norm: {total_param_change:.6f}")
        self.assertTrue(changed, f"Parameters did not update on step() for {optimizer_class.__name__}")
        print(f"✓ Parameters successfully updated for {optimizer_class.__name__}\n")

    @unittest.skipIf(not hasattr(torch, "compile"), "torch.compile not available")
    def test_step_updates_params(self):
        self._test_step_updates_params_helper(QUAD)
        self._test_step_updates_params_helper(Procrustes)

    def _test_loss_decreases_over_steps_helper(self, optimizer_class):
        print(f"\n=== Testing loss decrease over steps for {optimizer_class.__name__} ===")
        torch.manual_seed(0)

        model = torch.nn.Sequential(
            torch.nn.Linear(8, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 2),
        )

        optimizer = optimizer_class(model.parameters())

        x = torch.randn(64, 8)
        target = torch.zeros(64, 2)
        criterion = torch.nn.MSELoss()

        model.train()
        with torch.no_grad():
            initial_loss = criterion(model(x), target).item()
        print(f"Initial loss: {initial_loss:.6f}")

        initial_params = [p.detach().clone() for p in model.parameters()]
        
        steps = 5
        losses = [initial_loss]
        
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), target)
            loss.backward()
            
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            optimizer.step()
            
            with torch.no_grad():
                step_loss = criterion(model(x), target).item()
            losses.append(step_loss)
            
            print(f"Step {step + 1}: loss={step_loss:.6f}, grad_norm={total_grad_norm:.6f}, "
                  f"loss_change={step_loss - losses[step]:.6f}")

        final_loss = losses[-1]
        print(f"Final loss: {final_loss:.6f}")
        print(f"Total loss reduction: {initial_loss - final_loss:.6f} "
              f"({100 * (initial_loss - final_loss) / initial_loss:.2f}%)")

        total_param_change = 0.0
        for i, (param, param0) in enumerate(zip(model.parameters(), initial_params)):
            param_change = (param - param0).norm().item()
            total_param_change += param_change
            print(f"Param {i} total change norm: {param_change:.6f}")
        
        print(f"Total parameter change norm: {total_param_change:.6f}")

        self.assertLess(
            final_loss, initial_loss, f"Loss did not decrease over optimization steps for {optimizer_class.__name__}"
        )
        print(f"✓ Loss successfully decreased for {optimizer_class.__name__}\n")

    @unittest.skipIf(not hasattr(torch, "compile"), "torch.compile not available")
    def test_loss_decreases_over_steps(self):
        self._test_loss_decreases_over_steps_helper(QUAD)
        self._test_loss_decreases_over_steps_helper(Procrustes)


if __name__ == "__main__":
    unittest.main()
