function T = createTestImage(m,n,nframes,ex)

if ~exist('m','var'),       m       = 128;  end
if ~exist('n','var'),       n       = 128;  end
if ~exist('nframes','var'), nframes = 16;   end
if ~exist('ex','var'),      ex      = 1;    end

switch ex
    
    case 1
        
        [x,y]=meshgrid(linspace(-1,1,m),linspace(-1,1,n));
        s = 0.1;
        
        % rotation
        tt = linspace(0,2*pi,nframes+1);
        T = zeros(m,n,nframes);
        
        for i=1:numel(tt)-1
            t = tt(i);
            r  = 0.3;
            dx = r*cos(t);
            dy = r*sin(t);
            T(:,:,i) = (exp(- ((x+dx).^2 + (y+dy).^2) / (2*s) ));
        end
        
    case 2
        
        [x,y]=meshgrid(linspace(-1,1,m),linspace(-1,1,n));
        s = 0.1;
        
        % translation
        tt = linspace(-0.5,0.5,nframes);
        T = zeros(m,n,nframes);
        
        for i=1:numel(tt)
            t = tt(i);
            dx = t;
            dy = t;
            T(:,:,i) = (exp(- ((x+dx).^2 + (y+dy).^2) / (2*s) ));
        end
        
        
    case 3
        
        [x,y]=meshgrid(linspace(-1,1,m),linspace(-1,1,n));
        s1 = 0.1;
        s2 = 0.05;
        
        % translation + intesity change
        tt = linspace(-0.5,0.5,nframes);
        T = zeros(m,n,nframes);
        
        for i=1:numel(tt)
            t = tt(i);
            dx = t;
            dy = t;
            T(:,:,i) = exp(- ((x+dx).^2 + (y+dy).^2) / (2*s1) );
            M = exp(- ((x+dx).^2 + (y+dy).^2) / (2*s2) );
            T(:,:,i) = T(:,:,i) - rand * M .* T(:,:,i);
        end

    otherwise
        error('wrong ex ???')
        
end

if nargout == 0
    figure;
    colormap gray(256);
    for k=1:nframes
        imshow(T(:, :, k), [0, 1], 'InitialMagnification', 'fit');
        title(sprintf('T_{%d}',k));
        colorbar;
        waitforbuttonpress;
    end
end